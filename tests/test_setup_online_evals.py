import os
import unittest
from types import SimpleNamespace
from unittest import mock

from scripts import setup_online_evals as setup


class _Response:
    def __init__(self, status_code: int, body=None):
        self.status_code = status_code
        self._body = body
        self.text = ""

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class _FakeLangSmith:
    """Records the run-rule calls setup makes against the LangSmith REST API."""

    def __init__(self, secrets=()):
        self.secrets = list(secrets)
        self.deleted: list[str] = []
        self.created: list[dict] = []

    def get(self, url, headers=None, params=None):
        if url.endswith("/workspaces/current/secrets"):
            return _Response(200, [{"key": key} for key in self.secrets])
        if url.endswith("/playground-settings"):
            return _Response(200, [{"name": "claude-haiku-5-5", "id": "preset-1", "settings": {"lc": 1}}])
        if url.endswith("/runs/rules"):
            return _Response(200, [{"id": "rule-1", "display_name": "grounded"}])
        return _Response(404)

    def delete(self, url, headers=None):
        self.deleted.append(url)
        return _Response(204)

    def post(self, url, headers=None, json=None):
        self.created.append(dict(json or {}))
        return _Response(200, {})


class SetupOnlineEvalsTest(unittest.TestCase):
    def _run(self, judge_model: str, api: _FakeLangSmith, chat=None):
        env = SimpleNamespace(session_id="session-1", agent_id="agent", environment="production")
        with (
            mock.patch.dict(os.environ, {"LANGSMITH_API_KEY": "test-key"}),
            mock.patch("sys.argv", ["setup_online_evals"]),
            mock.patch.object(setup, "JUDGE_MODEL", judge_model),
            mock.patch.object(setup, "requests", api),
            mock.patch.object(setup, "Client"),
            mock.patch.object(setup, "_environment", return_value=env),
            mock.patch.object(setup, "ChatOpenAI", chat or mock.MagicMock()),
        ):
            setup.main()

    def test_custom_judge_preflight_success_recreates_rules(self):
        api = _FakeLangSmith()
        self._run("custom/claude-haiku-5-5", api)
        self.assertEqual(len(api.deleted), 1)
        self.assertEqual(len(api.created), len(setup.ONLINE_EVALUATORS))

    def test_custom_judge_preflight_failure_exits_before_touching_rules(self):
        api = _FakeLangSmith()
        chat = mock.MagicMock()
        chat.return_value.invoke.side_effect = RuntimeError("401 Unauthorized")
        with self.assertRaises(SystemExit) as cm:
            self._run("custom/claude-haiku-5-5", api, chat)
        self.assertNotIn(cm.exception.code, (0, None))
        self.assertEqual(api.deleted, [])
        self.assertEqual(api.created, [])

    def test_anthropic_judge_with_workspace_secret_recreates_rules(self):
        api = _FakeLangSmith(secrets=["ANTHROPIC_API_KEY"])
        self._run("anthropic/claude-haiku-5-5", api)
        self.assertEqual(len(api.deleted), 1)
        self.assertEqual(len(api.created), len(setup.ONLINE_EVALUATORS))

    def test_anthropic_judge_without_workspace_secret_exits_before_touching_rules(self):
        api = _FakeLangSmith(secrets=["OPENAI_API_KEY"])
        with self.assertRaises(SystemExit) as cm:
            self._run("anthropic/claude-haiku-5-5", api)
        self.assertNotIn(cm.exception.code, (0, None))
        self.assertIn("ANTHROPIC_API_KEY", str(cm.exception.code))
        self.assertEqual(api.deleted, [])
        self.assertEqual(api.created, [])

    def test_rules_score_only_agent_root_runs(self):
        api = _FakeLangSmith()
        self._run("custom/claude-haiku-5-5", api)
        self.assertTrue(api.created)
        for payload in api.created:
            self.assertEqual(payload["filter"], 'eq(name, "project_delivery_advisor")')


if __name__ == "__main__":
    unittest.main()
