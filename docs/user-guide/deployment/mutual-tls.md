# Mutual authentication between nodes and researcher

Fed-BioMed encrypts all gRPC traffic between nodes and the researcher with TLS. By
default only the researcher side is authenticated: each node trusts whichever
certificate the endpoint presents, and the researcher accepts any node that connects.
Mutual authentication makes the verification go both ways — the researcher requires and
verifies each node's client certificate, and each node pins the researcher's certificate
— so every channel is bound to a registered Fed-BioMed party.

The protocol carrying it is mTLS (mutual TLS), the variant of TLS where both sides
present a certificate. That is why gRPC failures around it are reported as TLS
handshake errors.

It is what establishes who the parties are, which is why a deployment cannot go without
it. Two situations make the point:

- **nothing else authenticates the parties** — Fed-BioMed running on plain Docker
  images, or embedded in an existing system that provides no identity of its own;
- **the surrounding infrastructure authenticates machines rather than parties**, so
  whoever is admitted to the network can still spoof another Fed-BioMed party at the
  gRPC level. Mutual authentication binds each channel to a registered component identity.

Either way it covers the "mutual verification of other party identity during gRPC setup"
item of the [security model](./security-model.md) (mitigating a malicious insider
man-in-the-middle that spoofs a party inside the network).

!!! warning "Mandatory for a deployment"
    `[authentication] force_mutual_authentication = True` **must be present in the
    configuration of the researcher and of every node** in a deployment. The setting is
    off by default, and an absent `[authentication]` section reads the same as
    `force_mutual_authentication = False` — a component left that way accepts an
    unverified peer, so it is not deployable.

    Leave it off only for local experimentation, where the channel falls back to the
    server-authenticated TLS described above and nothing verifies who the parties are.

## What mutual authentication changes

| | Without it (server-authenticated TLS) | Mutual authentication enabled |
|---|---|---|
| Node → researcher | Node fetches the server cert at connect and trusts it | Node **pins** a pre-registered researcher cert |
| Researcher → node | Node identity not checked | Node **must** present a registered client cert |
| Node identity | Declared in the message only | Every message's declared id must match the component id the presented certificate is registered under |
| New certificates | — | Picked up on the next handshake, **no restart** (hot-add) |
| `[authentication] force_mutual_authentication` | Absent or `False` — local experimentation only | `True`, **mandatory in a deployment**; read at startup, so turning it on or off takes effect only after **restarting** the component |

## Component certificates

Each component already owns a self-signed certificate, generated automatically when the
component is created (recorded in the `[certificate]` section of its config, under
`etc/`). The certificate's `CN=` (Common Name) field carries the component id
in the form `<NODE|RESEARCHER>_<uuid>`, which is what it identifies — an id, not an
organization. Mutual authentication reuses these certificates — you exchange and
register them, then flip a switch.

Certificates are role-restricted via Extended Key Usage: a node certificate is
`client`-only, a researcher certificate is `server`-only. A certificate carrying no role
restriction is accepted for both.

### Names on the researcher certificate

A node verifies the researcher by name, so the researcher certificate is issued for the
hosts nodes reach it at: the `[server] host` of its configuration, plus `localhost` and
`127.0.0.1`. For a researcher also reachable under a name its configuration does not
hold — a public DNS name, a second interface — issue it for those names too:

```bash
fedbiomed researcher certificate generate \
    --san fbm.hospital.org --san 10.0.0.9 --force
```

The component was issued a certificate when it was created, so this **requires
`--force`** — without it the command refuses to overwrite. The previous private key is
lost, so every party holding the old certificate has to register the new one.

## Enabling mutual authentication

Every component must already be created, since that is when its certificate is generated.
The setup is then symmetric: every party exports its certificate, shares it through a
trusted channel, and registers the certificates it receives. Then both sides turn mutual
authentication on.

### 1. Export and share each certificate

On every component (each node and the researcher), produce its registration bundle:

```shell
fedbiomed node       certificate registration-instructions   # on a node
fedbiomed researcher certificate registration-instructions   # on the researcher
```

Send the printed certificate to the other parties over a **trusted channel** (e-mail,
secure messaging, …). Fed-BioMed does not exchange certificates for you.

!!! note "Who registers whose certificate"
    - The **researcher** must register **every node's** certificate (the trust bundle
      it verifies clients against).
    - Each **node** must register the **researcher's** certificate (the one it pins).

    Node-to-node registration is not needed for mutual authentication.

### 2. Register the received certificates

Save each received certificate to a file and register it. The component id is read from the
certificate's `CN=` field, so `--component-id` is optional — supply it only for a third-party
certificate that carries no Fed-BioMed identity; given alongside an embedded identity, it
must match it:

```shell
# On the researcher: register each node certificate
fedbiomed researcher certificate register -pk /path/to/node1.pem
fedbiomed researcher certificate register -pk /path/to/node2.pem

# On each node: register the researcher certificate
fedbiomed node certificate register -pk /path/to/researcher.pem
```

Check what is registered at any time:

```shell
fedbiomed [node|researcher] certificate list   # shows component id, component type, expiry
```

Registration refuses combinations that cannot be valid:

- a component cannot register a certificate of its **own type** (checked against the
  component id and against the certificate's client/server role);
- a **node registers at most one certificate** — its researcher's. Registering a second
  component is rejected; re-registering the same component goes through `--upsert`.

`certificate list` reports a database that breaks either rule, and a node refuses to
start when several researcher certificates are registered, since it cannot tell which to
pin — delete the extras with `fedbiomed node certificate delete`.

### 3. Turn mutual authentication on

The config file (`etc/config.ini`) of **both** the researcher and every node must
carry, adding the section if it is not already there:

```ini
[authentication]
force_mutual_authentication = True
```

It is **mandatory in a deployment**: the entry has to be present and set to `True` on
every component. An absent `[authentication]` section reads the same as
`force_mutual_authentication = False`, so leaving it out silently leaves that component
unauthenticated rather than failing loudly.

Each component reads this setting when it starts, so change it while the component is
stopped, or restart it afterwards — on the researcher, by relaunching it or restarting
the Jupyter Notebook. Only the registered certificates are re-read while running, and a
component left on its previous setting behaves exactly as the
[state combinations](#mutual-authentication-enabled-or-disabled) below describe.

### 4. Start the components

Start the researcher first (it must have at least one node certificate registered or it
refuses to bind the port), then the nodes. On success you will see security log lines
such as:

- Researcher: `Node <NODE_id> identity verified by mutual authentication.`
- Node: `Mutually authenticated communication established with researcher …; node identity verified by the researcher.`

## Development / testing shortcut

When several components run locally under the same Fed-BioMed installation, one command
replaces steps 1 and 2: it registers in each of them the certificates it must trust — the
node certificates on the researcher, the researcher's on each node. It expects one
researcher and at least two nodes:

```shell
fedbiomed certificate-dev-setup
```

This registers certificates only; you still have to set
`[authentication] force_mutual_authentication = True` in each config, which remains
mandatory for any deployment. The command reads the local installation's components, so
it is for development and testing — it replaces no step of a real deployment.

## Scenarios

### Adding a node to a running instance (hot-add)

The researcher re-reads its trust bundle on every handshake, so a new node can join
**without restarting the researcher**:

1. On the new node: register the researcher certificate and set
   `[authentication] force_mutual_authentication = True`.
2. Send the new node's certificate to the researcher.
3. On the researcher: `fedbiomed researcher certificate register -pk /path/to/newnode.pem`.
4. Start the new node — it is trusted on its first connection.

### Renewing / rotating a certificate

Certificates are valid for 5 years. When one is renewed, its identity is unchanged but
the key and expiry differ, so every party that pinned or trusted it must register the
new one:

1. Regenerate the certificate on the component. Replacing an existing one requires
   `--force`, and the previous private key is lost:
   ```shell
   fedbiomed <component> certificate generate --force
   ```
2. Re-share it and re-register it on the other parties with `--upsert` to overwrite:
   ```shell
   fedbiomed researcher certificate register -pk /path/to/renewed.pem --upsert
   ```
3. The next handshake uses the new certificate; no restart of the researcher is needed
   for a renewed node certificate.

!!! info "Which expiries are watched"
    The researcher logs a warning when a registered node certificate is within 30 days
    of expiry (re-checked whenever the trust bundle changes). Nothing warns a node about
    its own certificate, nor about the researcher certificate it pins, and nothing warns
    the researcher about its own — check those with
    `fedbiomed [node|researcher] certificate list`, which prints every expiry date.

## State combinations

What each party ends up doing depends on both sides at once: whether mutual
authentication is enabled there, and what its certificate registry holds. The two tables
below give the outcome on each side for the combinations that occur in practice; the
[troubleshooting table](#verifying-and-troubleshooting) gives the fix for the failing
ones.

### Mutual authentication enabled or disabled

| Node `[authentication]` | Researcher `[authentication]` | On the node | On the researcher |
|---|---|---|---|
| off | off | Fetches the certificate the endpoint presents and trusts it: `Communication established … over server-authenticated TLS (node identity not verified)` | Accepts the node without checking its identity |
| on | on | Pins the registered researcher certificate and presents its own: `Mutually authenticated communication established …; node identity verified by the researcher` | `Node <NODE_id> identity verified by mutual authentication` on the node's first request |
| on | off | Stops on its first task request: `FB628 … the researcher does not verify node identities`, then `Node is stopped!`. The researcher names the node it authenticated in every task response; naming none is what tells the node | Sees the node's first request and answers it, then never hears from it again; it would have accepted it like any other, ignoring the client certificate presented |
| off | on | Stops: `FB628 … researcher requires mutual authentication but it is disabled on this node`, then `Node is stopped!`. No configuration the node can reach by retrying would connect | Rejects the node inside the TLS handshake; the node never appears, and the rejection is visible only with `GRPC_VERBOSITY=INFO` |

### Certificate state, with mutual authentication enabled on both sides

| Node state | Researcher state | On the node | On the researcher |
|---|---|---|---|
| Researcher certificate registered, own certificate valid | Node certificate registered, own certificate valid | Connects; node identity verified | Authenticates the node |
| Starts before its certificate is registered | Certificate registered while the researcher runs (hot-add) | Connects on a later retry, no restart | Picks the new certificate up at the next handshake, no restart |
| No researcher certificate registered | any | Refuses to start: `FB619 … no researcher certificate is registered` | The node never connects |
| Several researcher certificates registered | any | Refuses to start: `FB619 … researcher certificates are registered` — the one to pin is ambiguous | The node never connects |
| any | No node certificate registered | Endpoint never comes up; retries at debug level (`Researcher server is not available`) | Refuses to start: `FB619 … no node certificate is registered` |
| Certificate valid but not registered on the researcher | Trust bundle without that node | Retries; `FB628 … reachable but closes the connection during the TLS handshake`, logged once then at debug | Rejects it inside the handshake, no per-node log |
| Own certificate expired | Holds the expired certificate | Same as an unregistered certificate: handshake refused | Rejects it; `certificate_expiring` warnings had been raised from 30 days before that expiry |
| Own certificate regenerated with `--force`, not re-shared | Still holds the previous node certificate | Same as an unregistered certificate: handshake refused | Rejects it until the new certificate is registered with `--upsert` |
| Certificate within 30 days of expiry | Holds that certificate | Connects normally | Warning `NODE certificate <component_id> expires on <date>`, when the trust bundle is re-read |
| Pins an outdated or wrong researcher certificate | Serves its current certificate | Retries; `FB628 … Mutual authentication (mTLS) handshake with researcher failed` — treat as possible MITM | Handshake aborted by the node, nothing logged |
| Pins the researcher's current certificate | Own certificate expired | Same handshake failure as an outdated pin | No node can establish a channel |
| Declares a node id different from the component id its certificate is registered under | Certificate registered under that other component id | Stops: `FB628 … Researcher rejected this node's identity` | `FB628 … Declared node id … does not match the identity …`; the request is aborted `UNAUTHENTICATED` |
| Running with an established connection | Its certificate is deleted while the researcher runs | Stops on the next request; reconnection attempts are refused during the handshake | `FB628 … Refusing the node declaring id …: its certificate is not registered` |

## Verifying and troubleshooting

Both sides log the security state of the channel and record structured security audit
events, identifying the peer certificate by subject, issuer, serial and expiry — never
by its contents. The researcher additionally records the node's source address; it
re-records a node whose certificate or address changes, so reconnections and
certificate rotations are visible. A persisting failure is logged once at error level,
then repeated at debug level only until the connection recovers. If a connection does
not establish, match the symptom below — diagnosis is mostly **node-side**: the
researcher rejects untrusted nodes inside the TLS handshake and logs nothing per
rejected node (it says so once at startup).

!!! tip "Seeing rejected handshakes on the researcher"
    A rejection is reported by gRPC itself, at INFO. Fed-BioMed lowers gRPC to `ERROR`
    by default, because a node retrying its connection would otherwise fill the output
    with one line every couple of seconds. To watch rejections while diagnosing, start
    the researcher with `GRPC_VERBOSITY=INFO`, which takes precedence:

    ```shell
    GRPC_VERBOSITY=INFO fedbiomed researcher start
    ```

| Log / error | Cause | Fix |
|---|---|---|
| `FB619 … no researcher certificate is registered` (node won't start) | Mutual authentication on, researcher cert missing on node | Register the researcher certificate on the node |
| `FB628 … Mutual authentication (mTLS) handshake with researcher failed` (node retries) | Pinned researcher cert wrong/outdated, or possible MITM | Re-register the current researcher certificate on the node |
| `FB628 … reachable but closes the connection during the TLS handshake` (node retries) | Node cert not registered on the researcher — rejected inside the handshake | Register the node's certificate on the researcher |
| `FB628 … researcher requires mutual authentication but it is disabled on this node` (node stops) | Researcher has mutual authentication on, node has it off | Enable `[authentication]` on the node and register the researcher certificate there; register this node's certificate on the researcher side |
| `FB628 … Researcher rejected this node's identity` (node stops) | Declared node id ≠ the component id the node's certificate is registered under, or that certificate was deleted on the researcher | Ensure the node id matches how its certificate is registered, and that it is still registered |
| `FB628 … the researcher does not verify node identities: NO node in the federation is authenticated` (node stops) | Node has mutual authentication on, researcher has it off, so the researcher names no node in its task responses | Enable `[authentication]` on the researcher and register this node's certificate there. Disabling it on the node instead clears the error, but leaves the federation unauthenticated and is not deployable |
| `FB628 … verified this connection as <other id>` (node stops) | The researcher answered naming a component that is not this node. A researcher rejects a mismatched id before answering, so this is not a configuration case — treat it as a researcher that is not behaving as one | Investigate the endpoint the node is pinned to; a certificate registered under the wrong id surfaces as `Researcher rejected this node's identity` instead |
| `FB619 … no node certificate is registered` (researcher won't start) | Mutual authentication on, but no node certificate registered | Register at least one node certificate. Setting `[authentication] force_mutual_authentication = False` also starts the researcher, but unauthenticated — not an option for a deployment |
| `FB628 … Declared node id … does not match the identity … its certificate is registered under` (researcher error) | A node declared an id different from the one its certificate is registered under | Investigate; the node id and its registered component id must be the same component |
| `FB628 … Refusing the node declaring id … its certificate is not registered` (researcher error) | The node completed the handshake but its certificate is absent from the registry — typically deleted while the researcher was running | Re-register the node's certificate, or leave it rejected if the removal was deliberate |
| `FB628 … Refusing the node declaring id … its certificate registry could not be read` (researcher error) | The certificate database could not be read even once, so no node can be identified. Look for the accompanying `certificate_store_unreadable` warning | Check the path and permissions of the `db` entry in the researcher config |

!!! danger "A handshake failure may be an attack"
    Under mutual authentication a failed handshake is not silently retried as "server
    unavailable" — it is logged as a security event. If certificates are correct and a
    node still cannot connect, treat it as a possible man-in-the-middle and verify the
    endpoint.
