from typing import Any

import httpx
import requests

from task._constants import USER_SERVICE_ENDPOINT


class UserNotFoundError(httpx.HTTPStatusError):
    """Raised when a requested user no longer exists in the User Service."""


class UserClient:
    def get_all_users(self) -> list[dict[str, Any]]:
        headers = {"Content-Type": "application/json"}

        response = requests.get(
            url=USER_SERVICE_ENDPOINT + "/v1/users", headers=headers
        )

        if response.status_code == 200:
            data = response.json()
            print(f"Get {len(data)} users successfully")
            return data

        raise Exception(f"HTTP {response.status_code}: {response.text}")

    async def aget_all_users(self) -> list[dict[str, Any]]:
        """Asynchronously fetch all users for callers that run in an event loop."""
        headers = {"Content-Type": "application/json"}
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url=USER_SERVICE_ENDPOINT + "/v1/users", headers=headers
            )
            response.raise_for_status()
        data: list[dict[str, Any]] = response.json()
        print(f"Get {len(data)} users successfully")
        return data

    async def get_user(self, id: int) -> dict[str, Any]:
        """Asynchronously fetch one user, treating a 404 as an expected absence."""
        headers = {"Content-Type": "application/json"}
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url=f"{USER_SERVICE_ENDPOINT}/v1/users/{id}", headers=headers
            )
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as error:
                if response.status_code == 404:
                    raise UserNotFoundError(
                        f"User {id} was not found",
                        request=error.request,
                        response=error.response,
                    ) from error
                raise
        data: dict[str, Any] = response.json()
        return data

    def search_users(
        self,
        name: str | None = None,
        surname: str | None = None,
        email: str | None = None,
        gender: str | None = None,
    ) -> list[dict[str, Any]]:
        headers = {"Content-Type": "application/json"}

        # Only include parameters that are not None
        params = {}
        if name:
            params["name"] = name
        if surname:
            params["surname"] = surname
        if email:
            params["email"] = email
        if gender:
            params["gender"] = gender

        response = requests.get(
            url=USER_SERVICE_ENDPOINT + "/v1/users/search",
            headers=headers,
            params=params,
        )

        if response.status_code == 200:
            data = response.json()
            print(f"Get {len(data)} users successfully")
            return data

        raise Exception(f"HTTP {response.status_code}: {response.text}")

    def health(self):
        headers = {"Content-Type": "application/json"}

        response = requests.get(url=USER_SERVICE_ENDPOINT + "/health", headers=headers)

        if response.status_code == 200:
            data = response.json()
            return data

        raise Exception(f"HTTP {response.status_code}: {response.text}")
