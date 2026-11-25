import unittest
from unittest.mock import patch

import app as service


class LocalDevelopmentGuardTests(unittest.TestCase):
    def test_loopback_client_and_host_are_both_required(self):
        with service.app.test_request_context(
            "/",
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
            headers={"Host": "localhost:5000"},
        ):
            self.assertTrue(service.is_loopback_request())

        with service.app.test_request_context(
            "/",
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
            headers={"Host": "mediscan.example"},
        ):
            self.assertFalse(service.is_loopback_request())

        with service.app.test_request_context(
            "/",
            environ_overrides={"REMOTE_ADDR": "10.0.0.8"},
            headers={"Host": "localhost:5000"},
        ):
            self.assertFalse(service.is_loopback_request())

    def test_guard_is_disabled_by_default(self):
        with service.app.test_request_context(
            "/",
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
            headers={"Host": "127.0.0.1:5000"},
        ):
            with patch.object(service, "LOCAL_UPLOAD_ENABLED", False):
                response, status = service.local_development_guard()
                self.assertEqual(status, 403)
                self.assertEqual(response.get_json()["error"], "local_development_route_disabled")

    def test_guard_allows_enabled_loopback_development_request(self):
        with service.app.test_request_context(
            "/",
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
            headers={"Host": "localhost:5000"},
        ):
            with patch.object(service, "LOCAL_UPLOAD_ENABLED", True):
                self.assertIsNone(service.local_development_guard())


if __name__ == "__main__":
    unittest.main()
