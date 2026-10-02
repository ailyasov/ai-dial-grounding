import asyncio
import json
import logging
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from pydantic import BaseModel, Field, SecretStr

from task._constants import API_KEY, DIAL_URL
from task.user_client import UserClient, UserNotFoundError

logger = logging.getLogger(__name__)

# TODO: Info about app:
# HOBBIES SEARCHING WIZARD
# Searches users by hobbies and provides their full info in JSON format:
#   Input: `I need people who love to go to mountains`
#   Output:
#     ```json
#       "rock climbing": [{full user info JSON},...],
#       "hiking": [{full user info JSON},...],
#       "camping": [{full user info JSON},...]
#     ```
# ---
# 1. Since we are searching hobbies that persist in `about_me` section - we need to embed only user `id` and `about_me`!
#    It will allow us to reduce context window significantly.
# 2. Pay attention that every 5 minutes in User Service will be added new users and some will be deleted. We will at the
#    'cold start' add all users for current moment to vectorstor and with each user request we will update vectorstor on
#    the retrieval step, we will remove deleted users and add new - it will also resolve the issue with consistency
#    within this 2 services and will reduce costs (we don't need on each user request load vectorstor from scratch and pay for it).
# 3. We ask LLM make NEE (Named Entity Extraction) https://cloud.google.com/discover/what-is-entity-extraction?hl=en
#    and provide response in format:
#    {
#       "{hobby}": [{user_id}, 2, 4, 100...]
#    }
#    It allows us to save significant money on generation, reduce time on generation and eliminate possible
#    hallucinations (corrupted personal info or removed some parts of PII (Personal Identifiable Information)). After
#    generation we also need to make output grounding (fetch full info about user and in the same time check that all
#    presented IDs are correct).
# 4. In response we expect JSON with grouped users by their hobbies.
# ---
# This sample is based on the real solution where one Service provides our Wizard with user request, we fetch all
# required data and then returned back to 1st Service response in JSON format.
# ---
# Useful links:
# Chroma DB: https://docs.langchain.com/oss/python/integrations/vectorstores/index#chroma
# Document#id: https://docs.langchain.com/oss/python/langchain/knowledge-base#1-documents-and-document-loaders
# Chroma DB, async add documents: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.aadd_documents
# Chroma DB, get all records: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.get
# Chroma DB, delete records: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.delete
# ---
# TASK:
# Implement such application as described on the `flow.png` with adaptive vector based grounding and 'lite' version of
# output grounding (verification that such user exist and fetch full user info)

"""
## Detailed Task Breakdown

### Phase 1: Environment & Client Configuration
- [ ] **1.1 Configure Model Clients & Constants**
  - Configure `AzureChatOpenAI`:
    - Deployment: `gpt-4o`
    - Endpoint: `DIAL_URL`
    - API Key: `SecretStr(API_KEY)`
    - Temperature: `0.0`
    - API Version: `""` or `"2024-02-01"`
  - Configure `AzureOpenAIEmbeddings`:
    - Deployment: `text-embedding-3-small-1`
    - Endpoint: `DIAL_URL`
    - API Key: `SecretStr(API_KEY)`
    - Dimensions: `384`
    - `check_embedding_ctx_length=False`

### Phase 2: Data Models & Document Serialization
- [ ] **2.1 Define Pydantic Models for Structured LLM Output**
  - `GroupingResult`:
    - `hobby`: `str` (hobby name, e.g. "rock climbing", "hiking")
    - `user_ids`: `list[int]` (list of matching user IDs)
  - `GroupingResults`:
    - `grouping_results`: `list[GroupingResult]`
  - Output parser: `PydanticOutputParser(pydantic_object=GroupingResults)`
- [ ] **2.2 Implement Document Formatter**
  - Implement `format_user_document(user: dict[str, Any]) -> str`:
    - Serialize only `id` and `about_me` fields to minimize token usage and protect PII.
    - Format:
      ```text
      User:
        id: {id}
        About user: {about_me}
      ```

### Phase 3: Input Grounder Implementation
- [ ] **3.1 Asynchronous Context Management & VectorStore Initialization**
  - Implement `InputGrounder.__aenter__` and `initialize_vectorstore(batch_size=50)`:
    - Fetch all users from UserService via `await user_client.aget_all_users()`.
    - Create `Document(id=str(user["id"]), page_content=format_user_document(user))`.
    - Chunk documents into batches (e.g. 50 items) to respect embedding rate limits.
    - Instantiate `Chroma(collection_name="users", embedding_function=self.embeddings)`.
    - Asynchronously ingest batches with `asyncio.gather(*[vectorstore.aadd_documents(batch) ...])`.
- [ ] **3.2 Implement VectorStore Incremental Synchronization**
  - Implement `InputGrounder._update_vectorstore()`:
    - Query live users: `users = await self.user_client.aget_all_users()`.
    - Retrieve existing IDs in Chroma: `vectorstore.get()["ids"]`.
    - Compute diff:
      - `ids_to_delete = vectorstore_ids - current_user_ids`
      - `new_user_ids = current_user_ids - vectorstore_ids`
    - If `ids_to_delete`: call `vectorstore.delete(list(ids_to_delete))`.
    - If `new_user_ids`: prepare `Document` objects and ingest into Chroma in batches.
- [ ] **3.3 Implement Semantic Retrieval & Relevance Filtering**
  - Implement `InputGrounder.retrieve_context(query: str, k: int = 100, score: float = 0.2) -> str`:
    - Trigger `_update_vectorstore()` to ensure cache freshness.
    - Execute `await asimilarity_search_with_relevance_scores(query, k=k)`.
    - Collect matching documents' `page_content`.
    - Join contexts with `\n\n`.
- [ ] **3.4 Implement Prompt Augmentation & Generation**
  - Implement `InputGrounder.augment_prompt(query: str, context: str) -> str`:
    - Populate `USER_PROMPT` with `{context}` and `{query}`.
  - Implement `InputGrounder.generate_answer(augmented_prompt: str) -> GroupingResults`:
    - Construct `ChatPromptTemplate` with `SYSTEM_PROMPT` (partialed with `format_instructions`) and user message.
    - Execute LCEL pipeline: `await (prompt | llm_client | parser).ainvoke({})`.
    - Return validated `GroupingResults`.

### Phase 4: Output API Grounding Implementation
- [ ] **4.1 Implement Output Grounding Logic**
  - Implement `OutputGrounder._find_users(ids: list[int]) -> list[dict[str, Any]]`:
    - Concurrently fetch user profiles with `asyncio.gather` calling `user_client.get_user(user_id)`.
    - Catch 404 HTTP errors gracefully (log and return `None` for missing users).
    - Filter out `None` values to return only verified active users.
  - Implement `OutputGrounder.ground_response(grouping_results: GroupingResults)`:
    - Iterate through each `GroupingResult`.
    - Fetch full user profiles via `_find_users(grouping_result.user_ids)`.
    - Output/print grouped result mapping each hobby to full user JSON profiles.

### Phase 5: Application Entrypoint & Interactive Loop
- [ ] **5.1 Wire Up Main Loop**
  - Implement `main()`:
    - Initialize `AzureOpenAIEmbeddings` and `OutputGrounder`.
    - Open `async with InputGrounder(embeddings, llm_client) as rag:`.
    - Prompt user in loop (`> `).
    - On input:
      1. `context = await rag.retrieve_context(user_question)`
      2. `augmented_prompt = rag.augment_prompt(user_question, context)`
      3. `grouping_results = await rag.generate_answer(augmented_prompt)`
      4. `await output_grounder.ground_response(grouping_results)`
    - Handle termination commands (`quit`, `exit`).

### Phase 6: Verification & Validation
- [ ] **6.1 Verify UserService Connectivity**
  - Ensure mock user service container is running (`docker-compose up -d`).
  - Verify `/v1/users`, `/v1/users/{id}`, and `/health` endpoints.
- [ ] **6.2 Unit/Functional Verification**
  - Test cold start indexing in Chroma.
  - Test incremental update (add/remove users detection).
  - Verify semantic retrieval for queries like "I need people who love to go to mountains".
  - Verify Pydantic parsing of LLM output.
  - Verify output grounding correctly hydrates full user objects and discards invalid/404 IDs.
"""


SYSTEM_PROMPT = """
You are a RAG-powered assistant that assists users with their questions about user information.
Answer ONLY based on RAG context.
You are performing Named Entity Extraction (NEE) on the provided RAG context.
Your task is to identify hobbies mentioned in the context and return a JSON object where each hobby
maps to a list of user IDs who have that hobby.
- Use ONLY the provided RAG context to extract hobbies and user IDs.
- Do NOT invent or rewrite any personal data. If a user ID is not present in the context,
do not include it in the output.
The output must strictly adhere to the following JSON format:
{format_instructions}
"""

USER_PROMPT = """
## RAG CONTEXT:
{context}
User QUESTION:
{query}
"""

REPAIR_PROMPT = """
Your previous answer could not be parsed. Return the answer again using only the
required JSON schema and the supplied RAG context. Do not include prose,
Markdown fences, or any fields outside that schema.
"""


class GroupingResult(BaseModel):
    """Model for grouping result."""

    hobby: str = Field(..., description="Hobby name")
    user_ids: list[int] = Field(..., description="List of user IDs")


class GroupingResults(BaseModel):
    """Model for grouping results."""

    grouping_results: list[GroupingResult] = Field(
        ..., description="List of grouping results"
    )


def format_user_document(user: dict[str, Any]) -> str:
    """Format user document for embedding."""
    return f"User:\n  id: {user['id']}\n  About user: {user['about_me']}"


class InputGrounder:
    """Class for input grounding."""

    def __init__(self, embeddings: AzureOpenAIEmbeddings, llm_client: AzureChatOpenAI):
        self.embeddings = embeddings
        self.llm_client = llm_client
        self.vectorstore: Chroma | None = None
        self.user_client = UserClient()

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize_vectorstore()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""

    async def initialize_vectorstore(self, batch_size: int = 50):
        """Initialize vectorstore with all users."""
        print("🔎 Loading all users...")
        users = await self.user_client.aget_all_users()
        documents = [
            Document(id=str(user["id"]), page_content=format_user_document(user))
            for user in users
        ]
        self.vectorstore = Chroma(
            collection_name="users", embedding_function=self.embeddings
        )
        # Batch processing to respect embedding rate limits
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            await self.vectorstore.aadd_documents(batch)
        print("✅ Vectorstore is ready.")

    async def _update_vectorstore(self):
        """Update vectorstore with new and removed users."""
        if not self.vectorstore:
            raise ValueError("Vectorstore is not initialized.")
        current_users = await self.user_client.aget_all_users()
        current_user_ids = {str(user["id"]) for user in current_users}
        existing_ids = set((await asyncio.to_thread(self.vectorstore.get))["ids"])

        # Identify users to delete and add
        ids_to_delete = existing_ids - current_user_ids
        new_user_ids = current_user_ids - existing_ids

        # Delete removed users
        if ids_to_delete:
            await self.vectorstore.adelete(list(ids_to_delete))

        # Add new users
        if new_user_ids:
            new_users = [
                user for user in current_users if str(user["id"]) in new_user_ids
            ]
            documents = [
                Document(id=str(user["id"]), page_content=format_user_document(user))
                for user in new_users
            ]
            # Batch processing to respect embedding rate limits
            batch_size = 50
            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]
                await self.vectorstore.aadd_documents(batch)

    async def retrieve_context(
        self, query: str, k: int = 100, score: float = 0.2
    ) -> str:
        """Retrieve context from vectorstore based on query."""
        if not self.vectorstore:
            raise ValueError("Vectorstore is not initialized.")
        # Update vectorstore to ensure it's up-to-date
        await self._update_vectorstore()
        relevant_docs = await self.vectorstore.asimilarity_search_with_relevance_scores(
            query, k=k
        )
        context_parts = []
        for doc, relevance_score in relevant_docs:
            if relevance_score >= score:
                context_parts.append(doc.page_content)
                print(f"Score: {relevance_score}, Content: {doc.page_content}")
        return "\n\n".join(context_parts)

    def augment_prompt(self, query: str, context: str) -> str:
        """Augment user prompt with retrieved context."""
        augmented_prompt = USER_PROMPT.format(context=context, query=query)
        print(f"Augmented prompt:\n{augmented_prompt}")
        return augmented_prompt

    async def generate_answer(
        self, augmented_prompt: str, *, repair: bool = False
    ) -> GroupingResults:
        """Generate and validate structured output, optionally requesting a repair."""
        parser = PydanticOutputParser(pydantic_object=GroupingResults)
        if repair:
            augmented_prompt = f"{augmented_prompt}\n\n{REPAIR_PROMPT}"
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "{augmented_prompt}"),
            ]
        ).partial(format_instructions=parser.get_format_instructions())
        grouping_results: GroupingResults = await (
            prompt | self.llm_client | parser
        ).ainvoke(
            {"augmented_prompt": augmented_prompt}
        )
        return grouping_results


class OutputGrounder:
    """Class for output grounding."""

    def __init__(self, user_client: UserClient):
        self.user_client = user_client

    @staticmethod
    def _is_not_found_error(error: Exception) -> bool:
        """Return whether an HTTP error represents an expected missing user."""
        if isinstance(error, UserNotFoundError):
            return True

        response = getattr(error, "response", None)
        return getattr(response, "status_code", None) == 404

    async def _find_users(self, ids: list[int]) -> list[dict[str, Any]]:
        """Fetch active users, skipping malformed IDs and expected 404 responses."""
        valid_ids: list[int] = []
        for user_id in ids:
            # The Pydantic result model already requires integers. This second
            # check keeps this boundary safe if it is called directly or a
            # model is constructed outside normal validation.
            if isinstance(user_id, bool):
                logger.warning("Skipping invalid user ID %r", user_id)
                continue
            try:
                normalized_id = int(user_id)
            except (TypeError, ValueError):
                logger.warning("Skipping invalid user ID %r", user_id)
                continue
            valid_ids.append(normalized_id)

        tasks = [self.user_client.get_user(user_id) for user_id in valid_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        users: list[dict[str, Any]] = []
        for result in results:
            if isinstance(result, Exception):
                if self._is_not_found_error(result):
                    logger.info("User was deleted before output grounding: %s", result)
                    continue
                raise result
            if result is not None:
                users.append(result)
        return users

    async def ground_response(
        self, grouping_results: GroupingResults
    ) -> dict[str, list[dict[str, Any]]]:
        """Fetch, print as JSON, and return authoritative user profiles by hobby."""
        grounded_results: dict[str, list[dict[str, Any]]] = {}
        for grouping_result in grouping_results.grouping_results:
            users = await self._find_users(grouping_result.user_ids)
            grounded_results[grouping_result.hobby] = users
        print(json.dumps(grounded_results, ensure_ascii=False, indent=2))
        return grounded_results


async def main():
    embeddings = AzureOpenAIEmbeddings(
        deployment="text-embedding-3-small-1",
        azure_endpoint=DIAL_URL,
        openai_api_key=SecretStr(API_KEY),
        openai_api_version="2024-02-01",
        dimensions=384,
        check_embedding_ctx_length=False,
    )
    llm_client = AzureChatOpenAI(
        deployment_name="gpt-4o",
        openai_api_version="2024-02-01",
        azure_endpoint=DIAL_URL,
        openai_api_key=SecretStr(API_KEY),
        temperature=0.0,
    )

    output_grounder = OutputGrounder(user_client=UserClient())

    async with InputGrounder(embeddings, llm_client) as rag:
        while True:
            user_question = input("> ")
            if user_question.lower() in ["quit", "exit"]:
                break
            try:
                context = await rag.retrieve_context(user_question)
                if not context:
                    print("No matches found.")
                    continue

                augmented_prompt = rag.augment_prompt(user_question, context)
                try:
                    grouping_results = await rag.generate_answer(augmented_prompt)
                except OutputParserException:
                    logger.warning(
                        "Model response was malformed; retrying once with repair instructions."
                    )
                    grouping_results = await rag.generate_answer(
                        augmented_prompt, repair=True
                    )

                await output_grounder.ground_response(grouping_results)
            except OutputParserException:
                logger.warning("Model response was malformed after one repair attempt.")
                print("No matches found.")
            except Exception:
                logger.exception("Unable to process the user request.")
                print("Unable to process the request. Please try again.")


if __name__ == "__main__":
    asyncio.run(main())
