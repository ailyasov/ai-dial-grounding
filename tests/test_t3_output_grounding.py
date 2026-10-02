import asyncio
import unittest

import requests
from pydantic import ValidationError

from task.t3.in_out_grounding import GroupingResult, OutputGrounder
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
    def test_grouping_result_uses_integer_ids(self):
        result = GroupingResult(hobby="hiking", user_ids=["12", 13])

        self.assertEqual(result.user_ids, [12, 13])
        with self.assertRaises(ValidationError):
            GroupingResult(hobby="hiking", user_ids=["not-an-id"])

    def test_find_users_skips_bad_ids_and_deleted_users(self):
        client = FakeUserClient(
            {
                1: {"id": 1, "name": "Active"},
                2: UserNotFoundError("User 2 was not found"),
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


if __name__ == "__main__":
    unittest.main()
