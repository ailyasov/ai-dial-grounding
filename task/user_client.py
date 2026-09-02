from typing import Any

import requests

from task._constants import USER_SERVICE_ENDPOINT


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

    async def get_user(self, id: int) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}

        response = requests.get(
            url=f"{USER_SERVICE_ENDPOINT}/v1/users/{id}", headers=headers
        )

        if response.status_code == 200:
            data = response.json()
            return data

        raise Exception(f"HTTP {response.status_code}: {response.text}")

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
