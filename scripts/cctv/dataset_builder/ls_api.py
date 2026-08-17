"""ls_api.py — shared Label Studio API client for this directory's scripts
(create_ls_project.py, promote_to_trainset.py's --ls-project mode).

Auth: an API token, from --token or $LABEL_STUDIO_API_TOKEN. By default this
is treated as a Personal Access Token — get one from the Label Studio UI's
Account & Settings page. That page hands you a long-lived REFRESH token, not
something usable directly: LSClient exchanges it once for a short-lived
ACCESS token via POST /api/token/refresh/ and sends THAT as
'Authorization: Bearer <access>', re-exchanging automatically if a call 401s
mid-run (access tokens default to a 5-minute lifetime). If the server has
legacy tokens enabled instead (Label Studio 1.23 default: disabled — see
dataset_builder/README.md "Troubleshooting"), pass --legacy-token to send the
token as-is via 'Authorization: Token <token>' with no exchange.
"""
import os

import requests


class LSClient:
    """Authenticated requests to a Label Studio server.

    --legacy-token: send the given token as-is, 'Authorization: Token <token>'.
    Only works if the server's org has legacy tokens enabled (LS 1.23 default:
    disabled).

    Default: the given token is a Personal Access Token — a long-lived
    REFRESH token (Account & Settings hands you this directly; org TTL
    defaults to 200 years). It is not valid on its own against ordinary
    endpoints, only against /api/token/refresh/, which mints a short-lived
    (5 min default) ACCESS token; THAT is what rides as
    'Authorization: Bearer <access>'. A 401 triggers exactly one re-mint +
    retry (covers an access token expiring mid-run), then gives up.
    """

    def __init__(self, base_url, token, legacy):
        self.base_url = base_url.rstrip("/")
        self.legacy = legacy
        self.refresh_token = token
        self.access_token = token if legacy else self._refresh()

    def _refresh(self):
        resp = requests.post(f"{self.base_url}/api/token/refresh/",
                              json={"refresh": self.refresh_token}, timeout=30)
        if not resp.ok:
            raise SystemExit(
                f"error: could not exchange the API token for an access token via "
                f"/api/token/refresh/: {resp.status_code} {resp.text[:500]}\n"
                f"  If this is a legacy DRF token rather than a Personal Access "
                f"Token from Account & Settings, pass --legacy-token instead.")
        try:
            return resp.json()["access"]
        except (ValueError, KeyError):
            raise SystemExit(f"error: unexpected token-refresh response: {resp.text[:500]}")

    def _headers(self):
        scheme = "Token" if self.legacy else "Bearer"
        return {"Authorization": f"{scheme} {self.access_token}"}

    def request(self, method, path, **kwargs):
        resp = requests.request(method, f"{self.base_url}{path}", headers=self._headers(), **kwargs)
        if resp.status_code == 401 and not self.legacy:
            self.access_token = self._refresh()
            resp = requests.request(method, f"{self.base_url}{path}", headers=self._headers(), **kwargs)
        return resp


def add_auth_args(ap):
    """Add the --url/--token/--legacy-token flags shared by every script that
    talks to the Label Studio API. Call ls_client_from_args(args) to build the
    client after parsing."""
    ap.add_argument("--url", default=os.environ.get("LABEL_STUDIO_URL", "http://localhost:8080"),
                    help="Label Studio base URL (default: $LABEL_STUDIO_URL or http://localhost:8080)")
    ap.add_argument("--token", default=os.environ.get("LABEL_STUDIO_API_TOKEN"),
                    help="API token (default: $LABEL_STUDIO_API_TOKEN) — a Personal Access "
                         "Token by default (exchanged for an access token automatically), "
                         "or a legacy token with --legacy-token")
    ap.add_argument("--legacy-token", action="store_true",
                    help="Treat --token as a legacy DRF token and send it as-is via "
                         "'Authorization: Token <token>', with no /api/token/refresh/ "
                         "exchange. Only works if the server's org has legacy tokens "
                         "enabled (Label Studio 1.23 default: disabled)")


def ls_client_from_args(args):
    if not args.token:
        raise SystemExit("error: no API token; pass --token or set LABEL_STUDIO_API_TOKEN")
    return LSClient(args.url, args.token, args.legacy_token)
