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
    `[authentication] mutual_authentication = True` **must be present in the
    configuration of the researcher and of every node** in a deployment. The setting is
    off by default, and an absent `[authentication]` section reads the same as
    `mutual_authentication = False` — a component left that way accepts an
    unverified peer, so it is not deployable.

    Leave it off only for local experimentation, where the channel falls back to the
    server-authenticated TLS described above and nothing verifies who the parties are.

## What mutual authentication changes

| | Without it (server-authenticated TLS) | Mutual authentication enabled |
|---|---|---|
| Node → researcher | Node fetches the server cert at connect and trusts it | Node **pins** a pre-registered researcher cert |
| Researcher → node | Node identity not checked | Node **must** present a registered client cert |
| Node identity | Declared in the message only | Every message's declared id must match the component id the presented certificate is registered under |
| New certificates | — | On the researcher: picked up on the next handshake, **no restart** (hot-add). On a node: read once at startup, so a **restart** applies them |

## Component certificates

Each component already owns a self-signed certificate, generated automatically when the
component is created (recorded in the `[certificate]` section of its config, under
`etc/`). Its subject carries two fields:

- `CN=` (Common Name) holds the **component id**, which is what the certificate
  identifies — an id, not a host;
- `O=` (Organization) is **`Fed-BioMed`**, marking a certificate Fed-BioMed itself
  issued, so its `CN=` is read as a component id rather than as free text.

Mutual authentication reuses these certificates — you exchange and register them, then
flip a switch.

`O=Fed-BioMed` records which tooling issued a certificate, not that anything verified
it: a self-signed certificate asserts its own subject, so the field is descriptive and
nothing treats it as a security check. A certificate issued elsewhere — by your own CA —
carries whatever `CN=` that issuer chose, often a hostname, and its component id is
supplied when you register it.

Certificates are role-restricted via Extended Key Usage: a node certificate is
`client`-only, a researcher certificate is `server`-only. A certificate leaving the role
open — carrying both, or no Extended Key Usage at all, as certificates issued elsewhere
often do — is registered, but reported: nothing in it then states which side of the
connection it is meant for.

### Names on the researcher certificate

A node verifies the researcher by name, so the researcher certificate is issued for the
hosts nodes reach it at: the `[server] host` of its configuration, and the names given
with `--san`, and nothing else. For a researcher reachable under a name its
configuration does not hold — a public DNS name, a second interface — issue it for those
names too:

```bash
fedbiomed researcher certificate generate \
    --san fbm.example.org --san 10.0.0.9 --force
```

Each name is written as what it is: an address goes in as an `iPAddress` entry, a host
name as a `dNSName` one, and TLS never matches one against the other. Naming any
loopback form — `localhost`, `127.0.0.1`, `::1` — issues the certificate for all three,
since a node on the researcher's own machine dials whichever of them its configuration
holds.

A certificate stating no host is refused, and a node holding one does not start. TLS
falls back to the Common Name for such a certificate, and here the Common Name holds
the component id, never a host — so the node would be verifying the researcher against
a field that names no server. Only `dNSName` and `iPAddress` entries state a host: a
Subject Alternative Name carrying just an e-mail address or a URI, as one issued
outside Fed-BioMed may, states none. This applies whether the certificate is fetched
(server-authenticated TLS) or registered (mutual authentication).

The component was issued a certificate when it was created, so this **requires
`--force`** — without it the command refuses to overwrite. The previous private key is
lost, so every party holding the old certificate has to register the new one.

### When the address and the certificate name different hosts

A node dials the address in its `[researcher] ip`, and that address is what has to be
right: nothing in a certificate changes where the connection is made. The certificate
decides something else — the name TLS verifies the researcher under.

The two need not agree. Where the configured host is not among the names the
certificate carries, the node verifies the connection under the first name it does
carry, and **connects**. It reports the two values once at startup, at warning level:

```
The address this node creates the connection at is `10.9.9.9:50051`, read from
`[researcher] ip` and `[researcher] port` in its configuration. The researcher
certificate is issued for `researcher.example.org`, which does not include the host
`10.9.9.9`. The channel is verified under `researcher.example.org`, the first name the
certificate carries, rather than under the host dialled; the mismatch does not by
itself stop the node connecting.
```

Nothing else follows from it. What authenticates the researcher is the certificate the
node registered, not the address: a party answering at that address with any other
certificate is refused during the handshake. Two loopback forms of the same machine —
`127.0.0.1` configured against a certificate naming `localhost` — are not reported at
all.

!!! note "When the researcher moves to another address"
    Change `[researcher] ip` on the nodes and restart them. That is the whole
    procedure: the certificate names hosts, not the machine's current address, and
    keeps working. **Do not reissue the certificate for the new address** — `--force`
    destroys its private key, and every node then has to receive and register the new
    one before it can connect again. Reissue only on expiry, or if the key is
    compromised.

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
`CN=` field of a certificate carrying `O=Fed-BioMed`, so `--component-id` is optional —
supply it for a certificate issued elsewhere, whose `CN=` states no component id; given
alongside an embedded identity, it must match it:

```shell
# On the researcher: register each node certificate
fedbiomed researcher certificate register -pk /path/to/node1.pem
fedbiomed researcher certificate register -pk /path/to/node2.pem

# On each node: register the researcher certificate
fedbiomed node certificate register -pk /path/to/researcher.pem
```

Check what is registered at any time:

```shell
fedbiomed [node|researcher] certificate list   # shows component id and expiry
```

!!! note "On a node, the registry is read once — at startup"
    The researcher re-reads its trust bundle on every handshake, so a certificate
    registered or deleted there applies to the connections that follow. A node reads
    the researcher certificate it pins when it starts and keeps it for the whole run:
    registering, replacing (`--upsert`) or deleting a certificate on a node changes
    nothing for the running process — **restart the node** to apply it.

Registration refuses combinations that cannot be valid:

- a component cannot register a certificate restricted to its **own TLS role** — a node
  registers `server` certificates, a researcher `client` ones. A certificate that states
  neither is registered, with a warning naming the role that was expected;
- a **node registers at most one certificate** — its researcher's. Registering a second
  component is rejected; re-registering the same component goes through `--upsert`.

`certificate list` reports a node holding more than one certificate, and a node refuses
to start in that state, since it cannot tell which to pin — delete the extras with
`fedbiomed node certificate delete`.

### 3. Turn mutual authentication on

The config file (`etc/config.ini`) of **both** the researcher and every node must
carry, adding the section if it is not already there:

```ini
[authentication]
mutual_authentication = True
```

It is **mandatory in a deployment**: the entry has to be present and set to `True` on
every component, the researcher included.

Each component reads this setting when it starts, so change it while the component is
stopped, or restart it afterwards — on the researcher, by relaunching it or restarting
the Jupyter Notebook. Only the researcher re-reads its registered certificates while
running — a node keeps the certificate it read at startup — and a
component left on its previous setting behaves exactly as the
[state combinations](#by-the-mutual-authentication-setting) below describe.

### 4. Start the components

Start the researcher first (it must have at least one node certificate registered or it
refuses to bind the port), then the nodes. On success you will see security log lines
such as:

- Researcher: `Node <NODE_id> identity verified by mutual authentication.`
- Node: `Mutually authenticated communication established with researcher …; node identity verified by the researcher.`

The researcher retires connections once they reach a maximum age, so a node reconnects
regularly even while nothing goes wrong. The node announces its first connection and any
that follows an interruption; a connection merely replacing a retired one is logged at
debug level. The security audit log records every one of them.

## Operating a running federation

### Adding a node to a running instance (hot-add)

The researcher re-reads its trust bundle on every handshake, so a new node can join
**without restarting the researcher**:

1. On the new node: register the researcher certificate and set
   `[authentication] mutual_authentication = True`.
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
   A component reads its own certificate and private key when it starts, so restart it
   to present the new ones — a researcher included, whose own certificate is not part of
   what it re-reads while running.
2. Re-share it and re-register it on the other parties with `--upsert` to overwrite:
   ```shell
   fedbiomed researcher certificate register -pk /path/to/renewed.pem --upsert
   ```
3. The next handshake uses the new certificate; no restart of the researcher is needed
   for a renewed node certificate. A renewed **researcher** certificate is the other
   way round: every node has to re-register it and then be restarted, since a running
   node still pins the one it read at startup.

!!! info "Which expiries are watched"
    The researcher logs a warning when a registered node certificate is within 30 days
    of expiry (re-checked whenever the trust bundle changes). Nothing warns a node about
    its own certificate, nor about the researcher certificate it pins, and nothing warns
    the researcher about its own — check those with
    `fedbiomed [node|researcher] certificate list`, which prints every expiry date.

## Development and testing shortcut: certificate-dev-setup

When several components sit side by side in one directory, one command replaces steps 1
and 2: it registers in each of them the certificates it must trust — the node
certificates on the researcher, the researcher's on each node. It expects one researcher
and at least two nodes:

```shell
fedbiomed certificate-dev-setup
```

Components are read from the directory the command is run in, and only from its first
level — a component nested deeper is not part of the federation being set up. Give
`--path` to point at another directory:

```shell
fedbiomed certificate-dev-setup --path /path/to/components
```

Certificates already registered are reported and left as they are, so the command can be
re-run to complete a federation a node was added to. A registration that no longer
matches the certificate the component serves — after regenerating one, for instance —
fails the run instead, since it would leave a federation that cannot handshake.

`--prune` clears every registration in each component before writing the ones found
under the path, so the trust stores end up describing the components currently there and
nothing else. Use it after replacing or removing a component: without it the departed
component stays trusted, and a node left holding two researcher certificates cannot tell
which one to pin.

```shell
fedbiomed certificate-dev-setup --prune
```

This registers certificates only. Setting
`[authentication] mutual_authentication = True` in each config remains mandatory
for any deployment, and `--enable-mutual-authentication` does it for every component at
once, after their certificates are registered:

```shell
fedbiomed certificate-dev-setup --enable-mutual-authentication
```

Components already running have to be restarted to pick the setting up, and a running
node equally to pick up the certificates the command registered or pruned in it. The
command reads components that are already on the machine, so it is for development and
testing — it replaces no step of a real deployment.

## Outcome by state

What each party ends up doing depends on both sides at once: whether mutual
authentication is enabled there, and what its certificate registry holds. The two tables
below answer *what happens in this state*, for the combinations that occur in practice.
The message a failing state produces, and its fix, are indexed the other way round — by
what is logged — in [verifying and troubleshooting](#verifying-and-troubleshooting).

### By the mutual authentication setting

| Node `[authentication]` | Researcher `[authentication]` | On the node | On the researcher |
|---|---|---|---|
| off | off | Fetches the certificate the endpoint presents and trusts it: `Communication established … over server-authenticated TLS (node identity not verified)` | Accepts the node without checking its identity |
| on | on | Pins the registered researcher certificate and presents its own: `Mutually authenticated communication established …; node identity verified by the researcher` | `Node <NODE_id> identity verified by mutual authentication` on the node's first request |
| on | off | Warns, refuses the task and retries. The researcher names the node it authenticated in every task response; naming none is what tells the node. Enabling it on the researcher connects the node with no restart | Answers the request, then sees the node close the channel and ask again; it would have accepted it like any other, ignoring the client certificate presented |
| off | on | Stops. No configuration the node can reach by retrying would connect | Rejects the node inside the TLS handshake; the node never appears, and the rejection is visible only with `GRPC_VERBOSITY=INFO` |

### By certificate state (both sides enabled)

| Node state | Researcher state | On the node | On the researcher |
|---|---|---|---|
| Researcher certificate registered, own certificate valid | Node certificate registered, own certificate valid | Connects; node identity verified | Authenticates the node |
| Starts before its certificate is registered | Certificate registered while the researcher runs (hot-add) | Connects on a later retry, no restart | Picks the new certificate up at the next handshake, no restart |
| No researcher certificate registered | any | Refuses to start | The node never connects |
| Several researcher certificates registered | any | Refuses to start: the one to pin is ambiguous | The node never connects |
| any | No node certificate registered | Endpoint never comes up; retries at debug level | Refuses to start |
| No valid certificate of this node on the researcher: never registered, expired, or regenerated with `--force` and not re-shared | Trust bundle holding no current certificate for that node | Warns once, then retries at debug level. Registering the current certificate connects the node with no restart | Rejects it inside the handshake, no per-node log. An expiry had been announced by `certificate_expiring` warnings from 30 days before; a regenerated certificate is registered with `--upsert` |
| Certificate within 30 days of expiry | Holds that certificate | Connects normally | Warns when the trust bundle is re-read |
| Pins an outdated or wrong researcher certificate | Serves its current certificate | Warns once, then retries at debug level — treat as possible MITM | Handshake aborted by the node, nothing logged |
| Pins the researcher's current certificate | Own certificate expired | Same handshake failure as an outdated pin | No node can establish a channel |
| Declares a node id different from the component id its certificate is registered under | Certificate registered under that other component id | Warns once, then retries at debug level | Aborts the request `UNAUTHENTICATED` |
| Running with an established connection | Its certificate is deleted while the researcher runs | Rejected on the next request, then retries; reconnection attempts are refused during the handshake until it is registered again | Refuses the node until it is registered again |
| Running with an established connection | Restarted | Retries until the researcher answers again, at debug level; catching the restart window warns once | Authenticates the node again on its next request |
| Running while its own registry changes: the researcher certificate is registered, replaced or deleted on the node | any | Unaffected — it keeps pinning the certificate it read at startup. The change applies when the node is restarted | Unaffected |

## Verifying and troubleshooting

Both sides log the security state of the channel and record structured security audit
events, identifying the peer certificate by subject, issuer, serial and expiry — never
by its contents. The researcher additionally records the node's source address; it
re-records a node whose certificate or address changes, so reconnections and
certificate rotations are visible. A failure the node retries is logged once as a
warning, then repeated at debug level only until the connection recovers; only a state
that stops the node is an error. If a connection does
not establish, match the symptom below — diagnosis is mostly **node-side**: the
researcher rejects untrusted nodes inside the TLS handshake and logs nothing per
rejected node (it says so once at startup).

!!! tip "Seeing failed handshakes as gRPC reports them"
    Each end reports its own side of a failed handshake through gRPC itself, at INFO,
    naming the OpenSSL reason: the researcher a node reached without a certificate logs
    `PEER_DID_NOT_RETURN_A_CERTIFICATE`, the node that cannot verify the researcher
    certificate it pinned logs `CERTIFICATE_VERIFY_FAILED`. Fed-BioMed lowers gRPC to
    `ERROR` on both components, because a node retrying its connection would otherwise
    add one line to each output every couple of seconds. A value set in the environment
    takes precedence, so raise it on the side being diagnosed:

    ```shell
    GRPC_VERBOSITY=INFO fedbiomed researcher start
    GRPC_VERBOSITY=INFO fedbiomed node start
    ```

| Log / error | Cause | Fix |
|---|---|---|
| `FB619 … no researcher certificate is registered` (node won't start) | Mutual authentication on, researcher cert missing on node | Register the researcher certificate on the node |
| `FB619 … certificates are registered` (node won't start) | More than one certificate is registered on the node, so the one to pin is ambiguous | Delete the extras with `fedbiomed node certificate delete`, keeping its researcher's |
| `FB628 … states no host` (node stops) | The researcher certificate names no host, so TLS would verify it on its Common Name, which holds a component id. Its Subject Alternative Name is absent, or carries only entries that name no server — an e-mail address, a URI. A certificate issued elsewhere, or one from a Fed-BioMed release that did not write a SAN | Request the researcher to reissue its certificate for the hosts nodes reach it at; under mutual authentication, register the new one on the node and restart it |
| Warning `The address this node creates the connection at is …` (node connects) | The configured `[researcher] ip` is not among the names the certificate carries, so the connection is verified under a name that is. Expected wherever a researcher is reached at an address its certificate does not name | Nothing is required — the node connects and the researcher is still authenticated by its certificate. To stop reporting it, have the researcher reissue for the address nodes dial, or set `[researcher] ip` to a name the certificate carries |
| `FB628 … does not carry the name this node verifies it under` (node stops) | The peer failed the name check against a name read from the registered certificate, so it is serving a different certificate — or one stating a name TLS cannot match | Register the researcher's current certificate on the node and restart it. If it is already current, request the researcher to reissue it for the hosts nodes reach it at |
| Debug `Researcher server is not available` (node retries) | The endpoint is not up. Under mutual authentication a researcher with no node certificate registered refuses to start, so this is what a node sees of it | Check the researcher started, and that at least one node certificate is registered there |
| Warning `Mutual authentication (mTLS) handshake with researcher failed` (node retries) | The certificates the two sides hold for each other do not match — most often a pinned researcher certificate that is wrong or outdated — or a possible MITM | Re-register the current researcher certificate on the node and restart it — a running node keeps the certificate it read at startup — and check this node's certificate is the one registered on the researcher |
| Warning `… reachable but closes the connection during the TLS handshake` (node retries) | Node cert not registered on the researcher, or expired — rejected inside the handshake. A researcher that is restarting closes connections the same way, and clears on its own | Register the node's certificate on the researcher; if it is registered, check its expiry, then whether the researcher was restarting |
| `FB628 … researcher requires mutual authentication but it is disabled on this node` (node stops) | Researcher has mutual authentication on, node has it off | Enable `[authentication]` on the node and register the researcher certificate there, then request the researcher to register this node's certificate |
| Warning `Researcher rejected this node's identity` (node retries) | Its certificate is not registered on the researcher, or the node id it declares is not the one that certificate is registered under | Register the node's certificate on the researcher — the node connects with no restart — and ensure the node id matches how it is registered |
| Warning `… the researcher does not verify node identities: NO node in the federation is authenticated` (node retries) | Node has mutual authentication on, researcher has it off, so the researcher names no node in its task responses. The node refuses every task until that changes | Request the researcher to enable `[authentication]` and register this node's certificate — the node connects with no restart. Disabling it on the node instead also clears the warning, but leaves the federation unauthenticated and is not deployable |
| `FB628 … verified this connection as <other id>` (node stops) | The researcher answered naming a component that is not this node. A researcher rejects a mismatched id before answering, so this is not a configuration case — treat it as a researcher that is not behaving as one | Investigate the endpoint the node is pinned to; a certificate registered under the wrong id surfaces as `Researcher rejected this node's identity` instead |
| `FB619 … no node certificate is registered` (researcher won't start) | Mutual authentication on, but no node certificate registered | Register at least one node certificate. Setting `[authentication] mutual_authentication = False` also starts the researcher, but unauthenticated — not an option for a deployment |
| Warning `Certificate <component_id> expires on <date>` (researcher) | A registered node certificate is within 30 days of expiry; re-checked whenever the trust bundle changes | Renew it on that node, share it again, and re-register it with `--upsert` before that date |
| `FB628 … Declared node id … does not match the identity … its certificate is registered under` (researcher error) | A node declared an id different from the one its certificate is registered under | Investigate; the node id and its registered component id must be the same component |
| `FB628 … Refusing the node declaring id … its certificate is not registered` (researcher error) | The node completed the handshake but its certificate is absent from the registry — typically deleted while the researcher was running | Re-register the node's certificate, or leave it rejected if the removal was deliberate |
| `FB628 … Refusing the node declaring id … its certificate registry could not be read` (researcher error) | The certificate database could not be read even once, so no node can be identified. Look for the accompanying `certificate_store_unreadable` warning | Check the path and permissions of the `db` entry in the researcher config |

!!! danger "A handshake failure may be an attack"
    Under mutual authentication a failed handshake is not silently retried as "server
    unavailable" — it is logged as a security event. If certificates are correct and a
    node still cannot connect, treat it as a possible man-in-the-middle and verify the
    endpoint.
