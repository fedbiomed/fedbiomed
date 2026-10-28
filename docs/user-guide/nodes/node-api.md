# Node HTTP API

The API-only installation exposes node management services without a frontend.
It uses the same routes, authentication and databases as the Node GUI.

## Install and start

```sh
pip install "fedbiomed[node-api]"
fedbiomed-node-api --path /path/to/node
# Equivalent core command:
fedbiomed node --path /path/to/node api start
```

The default address is `http://localhost:8484`. `--path` defaults to `fbm-node`
in the current directory; a missing component is initialized. The data directory
defaults to `data` in the node root. Use `--data-folder /path/to/datasets` to
select an existing directory. These paths belong to the server host.

Starting the HTTP server does not start the federated-learning node. Use
`fedbiomed node --path /path/to/node start`, or the authenticated `/api/node/start`
endpoint. See the GUI guide for [HTTPS options](node-gui.md#https-configuration),
[upgrade notes](node-gui.md#upgrading-from-the-combined-package), and
[troubleshooting](../../support/troubleshooting.md#node-api-and-gui).

## API reference and OpenAPI file

The [complete API reference](node-api-reference.md) lists methods, paths,
authentication, JSON request schemas, examples and response formats.
[Download openapi.json](openapi.json) for client tools such as Postman.

Each installed API package also includes the same file and serves it without
authentication at `/openapi.json`, including when launched with the GUI:

```sh
curl -fsS http://localhost:8484/openapi.json -o openapi.json
```

Use your running server's file when its version differs from the documentation
website. The schema describes the current API, including legacy action-based
methods: for example, dataset listing and removal both use **POST**.

## Authenticate

The examples below use Bash and `jq` to read JSON responses.
Use the administrator account configured for your node. A fresh installation
uses `admin@fedbiomed.gui` / `admin` unless the initial credentials were changed
before first startup. Change that password through `/api/update-password` or the
GUI. See [administrator configuration](node-gui.md#default-admin-configuration).

```sh
BASE=http://localhost:8484/api
LOGIN=$(curl -fsS "$BASE/auth/token/login" \
  -H 'Content-Type: application/json' \
  -d '{"email":"admin@fedbiomed.gui","password":"admin"}')
TOKEN=$(printf '%s' "$LOGIN" | jq -er '.result.access_token')
REFRESH_TOKEN=$(printf '%s' "$LOGIN" | jq -er '.result.refresh_token')

curl -fsS "$BASE/config/node-id" -H "Authorization: Bearer $TOKEN"
```

Protected requests require `Authorization: Bearer <access_token>`.
Successful JSON responses normally contain `success`, `result`, and `message`.
Check HTTP status as well: rejected credentials or tokens return 401, and
administrator-only actions return 403 for a non-admin account. Framework errors
can have a different format; log downloads return plain text.

Access tokens expire after 30 minutes by default. Refresh with the **refresh
token**, then replace both stored tokens with the returned values:

```sh
LOGIN=$(curl -fsS "$BASE/auth/token/refresh" \
  -H "Authorization: Bearer $REFRESH_TOKEN")
TOKEN=$(printf '%s' "$LOGIN" | jq -er '.result.access_token')
REFRESH_TOKEN=$(printf '%s' "$LOGIN" | jq -er '.result.refresh_token')
```

If refresh fails or expires, log in again. The legacy logout route does not
revoke bearer tokens; discard tokens in the client when finished.

## Register and list datasets

List registrations (send an empty JSON object for no filter):

```sh
curl -fsS "$BASE/datasets/list" \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' -d '{}'
```

Register MNIST; the node downloads it if needed:

```sh
curl -fsS "$BASE/datasets/add-default-dataset" \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"name":"MNIST dataset","tags":["#mnist"],"desc":"MNIST handwritten digits"}'
```

Omitting `path` uses `defaults/mnist` below the selected data directory.
To choose an existing folder, add `"path":["my-mnist"]`. The array contains
directory segments relative to the server's data directory, not a client path.
For CSV, images or custom datasets, use `/api/datasets/add` with `type`, `name`,
`path`, `tags` and `desc`, as shown in the reference. Dataset removal uses
`POST /api/datasets/remove` with `{"dataset_id":"dataset_REPLACE_ME"}`; it
removes the registration, not the underlying files.

User creation/removal and configuration updates are documented in the reference
under `/api/admin/users` and `/api/node/config`. Read the current configuration
first: its field metadata identifies editable settings and their types.

## Use Postman

1. Download `/openapi.json` from your server, then use **Import** in Postman.
   Import it as a collection, or generate a collection from the specification.
   See [Postman's OpenAPI import documentation](https://learning.postman.com/docs/integrations/available-integrations/working-with-openAPI/).
2. Set the generated server/base URL variable to `http://localhost:8484`
   (no `/api` suffix; the paths already contain it). Adjust the scheme, host
   and port for your server. You can also set the request URLs directly.
3. Send `POST /api/auth/token/login` with **No Auth** and your email/password
   in a raw JSON body. Copy `result.access_token` into an environment variable
   named `token`; copy `result.refresh_token` into `refresh_token`.
4. For protected requests, select **Bearer Token** with `{{token}}`. You can
   configure this on the collection and select **Inherit auth from parent**
   on requests. Imported request-level auth may need replacing. Keep login,
   registration and `/openapi.json` on **No Auth**; use `{{refresh_token}}`
   for the refresh request.
5. Replace sample dataset paths, IDs and credentials before sending requests.
   Select **Body → raw → JSON** for JSON bodies, including DELETE user removal.

Optionally add this Post-response script to the login and refresh requests to
save their tokens automatically in the selected environment:

```javascript
if (pm.response.code === 200) {
  const result = pm.response.json().result;
  pm.environment.set("token", result.access_token);
  pm.environment.set("refresh_token", result.refresh_token);
}
```

Re-import the specification after upgrading the API. The OpenAPI file is the
maintained contract; a separate Postman collection is not required.
