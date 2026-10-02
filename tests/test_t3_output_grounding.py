import asyncio
import io
import unittest
from contextlib import redirect_stdout

import httpx
import requests
from pydantic import ValidationError

from task.t3.in_out_grounding import GroupingResult, GroupingResults, OutputGrounder
from task.user_client import UserNotFoundError


class FakeUserClient:
    def __init__(self, responses):
        self.responses = responses
        self.requested_ids = []

    async def get_user(self, user_id):
        self.requested_ids.append(user_id)
        response = self.responses[user_id]
        if isinstance(response, Exception):
            raise response
        return response


class OutputGrounderTests(unittest.TestCase):
    @staticmethod
    def _not_found_error(user_id):
        request = httpx.Request("GET", f"http://users.test/v1/users/{user_id}")
        response = httpx.Response(404, request=request)
        return UserNotFoundError(
            f"User {user_id} was not found", request=request, response=response
        )

    def test_grouping_result_uses_integer_ids(self):
        result = GroupingResult(hobby="hiking", user_ids=["12", 13])

        self.assertEqual(result.user_ids, [12, 13])
        with self.assertRaises(ValidationError):
            GroupingResult(hobby="hiking", user_ids=["not-an-id"])

    def test_find_users_skips_bad_ids_and_deleted_users(self):
        client = FakeUserClient(
            {
                1: {"id": 1, "name": "Active"},
                2: self._not_found_error(2),
            }
        )
        grounder = OutputGrounder(client)

        users = asyncio.run(grounder._find_users([1, "not-an-id", True, 2]))

        self.assertEqual(users, [{"id": 1, "name": "Active"}])
        self.assertEqual(client.requested_ids, [1, 2])

    def test_find_users_propagates_service_failures(self):
        error = requests.HTTPError("HTTP 503")
        client = FakeUserClient({1: error})

        with self.assertRaisesRegex(requests.HTTPError, "503"):
            asyncio.run(OutputGrounder(client)._find_users([1]))

    def test_ground_response_returns_and_prints_json(self):
        client = FakeUserClient({1: {"id": 1, "name": "Ada"}})
        grouping_results = GroupingResults(
            grouping_results=[GroupingResult(hobby="hiking", user_ids=[1])]
        )
        output = io.StringIO()

        with redirect_stdout(output):
            result = asyncio.run(
                OutputGrounder(client).ground_response(grouping_results)
            )

        self.assertEqual(result, {"hiking": [{"id": 1, "name": "Ada"}]})
        self.assertEqual(
            output.getvalue(),
            '{\n  "hiking": [\n    {\n      "id": 1,\n      "name": "Ada"\n    }\n  ]\n}\n',
        )


if __name__ == "__main__":
    unittest.main()
