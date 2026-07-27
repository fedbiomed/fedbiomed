# Mutual TLS (mTLS) between nodes and researcher

Fed-BioMed encrypts all gRPC traffic between nodes and the researcher with TLS. By
default the node authenticates the *researcher* (server-authenticated TLS), but the
researcher does **not** verify the identity of the connecting node. Mutual TLS (mTLS)
adds that missing direction: the researcher requires and verifies each node's client
certificate, and each node pins the researcher's certificate instead of trusting
whatever certificate the endpoint presents.

This directly addresses the "mutual verification of other party identity during gRPC
setup" item of the [security model](./security-model.md) (mitigating a malicious
insider man-in-the-middle that spoofs a party inside the network).

!!! info "mTLS is optional and off by default"
    When mTLS is disabled, the node fetches and trusts the researcher certificate on
    connect, and the researcher accepts any node. Nothing below is required unless you
    explicitly enable mTLS.

## What mTLS changes

| | Default (server-auth TLS) | Mutual TLS enabled |
|---|---|---|
| Node → researcher | Node fetches the server cert at connect and trusts it | Node **pins** a pre-registered researcher cert |
| Researcher → node | Node identity not checked | Node **must** present a registered client cert |
| Node identity | Declared in the message only | Every message's declared id must match the party id the presented certificate is registered under |
| New certificates | — | Picked up on the next handshake, **no restart** (hot-add) |

Each component already owns a self-signed certificate, generated automatically when the
component is created (recorded in the `[certificate]` section of its config, under
`etc/`). The certificate's `O=` (Organization) field carries the component's party id
in the form `<NODE|RESEARCHER>_<uuid>`. mTLS reuses these certificates — you exchange
and register them, then flip a switch.

Certificates are role-restricted via Extended Key Usage: a node certificate is
`client`-only, a researcher certificate is `server`-only. A certificate carrying no role
restriction is accepted for both.

## Enabling mTLS

The setup is symmetric: every party exports its certificate, shares it through a trusted
channel, and registers the certificates it receives. Then both sides turn mTLS on.

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

    Node-to-node registration is not needed for mTLS.

### 2. Register the received certificates

Save each received certificate to a file and register it. The party id is read from the
certificate's `O=` field, so `--party-id` is optional — supply it only for a third-party
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
fedbiomed [node|researcher] certificate list   # shows party id, component, expiry date
```

Registration refuses combinations that cannot be valid:

- a component cannot register a certificate of its **own type** (checked against the
  party id and against the certificate's client/server role);
- a **node registers at most one certificate** — its researcher's. Registering a second
  party is rejected; re-registering the same party goes through `--upsert`.

`certificate list` reports a database that breaks either rule, and a node refuses to
start when several researcher certificates are registered, since it cannot tell which to
pin — delete the extras with `fedbiomed node certificate delete`.

### 3. Turn mTLS on

The config file (`etc/config-*.ini`) of **both** the researcher and every node must
carry, adding the section if it is not already there:

```ini
[mtls]
enabled = True
```

An absent `[mtls]` section reads the same as `enabled = False`, so mTLS stays off until
the section says otherwise.

### 4. Start the components

Start the researcher first (it must have at least one node certificate registered or it
refuses to bind the port), then the nodes. On success you will see security log lines
such as:

- Researcher: `Node <NODE_id> authenticated via mutual TLS.`
- Node: `Mutual-TLS communication established with researcher …; node identity verified by the researcher.`

## Scenarios

### Standard deployment (1 researcher, N nodes)

1. Create all components (certificates are generated automatically).
2. Exchange certificates (step 1) through a trusted channel.
3. On the researcher, register **all** node certificates; on each node, register the
   researcher certificate (step 2).
4. Set `[mtls] enabled = True` everywhere (step 3).
5. Start researcher, then nodes (step 4).

### Adding a node to a running instance (hot-add)

The researcher re-reads its trust bundle on every handshake, so a new node can join
**without restarting the researcher**:

1. On the new node: register the researcher certificate and set `[mtls] enabled = True`.
2. Send the new node's certificate to the researcher.
3. On the researcher: `fedbiomed researcher certificate register -pk /path/to/newnode.pem`.
4. Start the new node — it is trusted on its first connection.

### Renewing / rotating a certificate

Certificates are valid for 5 years. When one is renewed, its identity is unchanged but
the key and expiry differ, so every party that pinned or trusted it must register the
new one:

1. Regenerate the certificate on the component. Replacing an existing one needs
   `--force`, since the previous private key is lost:
   ```shell
   fedbiomed <component> certificate generate --force
   ```
2. Re-share it and re-register it on the other parties with `--upsert` to overwrite:
   ```shell
   fedbiomed researcher certificate register -pk /path/to/renewed.pem --upsert
   ```
3. The next handshake uses the new certificate; no restart of the researcher is needed
   for a renewed node certificate.

!!! info "Expiry warnings"
    The researcher logs a warning when a registered node certificate is within 30 days
    of expiry (re-checked whenever the trust bundle changes). `certificate list` shows
    each certificate's expiry date.

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
| `FB619 … no node certificate is registered` (researcher won't start) | mTLS on, but no node certificate registered | Register at least one node certificate, or set `[mtls] enabled = False` |
| `FB628 … Mutual-TLS handshake with researcher failed` (node retries) | Pinned researcher cert wrong/outdated, or possible MITM | Re-register the current researcher certificate on the node |
| `FB628 … reachable but closes the connection during the TLS handshake` (node retries) | Node cert not registered on the researcher — rejected inside the handshake | Register the node's certificate on the researcher |
| `FB628 … Researcher rejected this node's identity` (node stops) | Declared node id ≠ the party id the node's certificate is registered under, or that certificate was deleted on the researcher | Ensure the node id matches how its certificate is registered, and that it is still registered |
| `node identity will NOT be verified` (node warning) | Node has mTLS on, researcher has it off — node connects anyway | Enable `[mtls]` on the researcher too |
| `could not determine whether the researcher verifies node identity` (node info) | Node has mTLS on; the check of whether the researcher demands client certificates was inconclusive (typically a busy or slow endpoint). The channel is up and the researcher certificate is pinned | Nothing to do; confirm on the researcher side that `[mtls] enabled = True` if you expect enforcement |
| `FB628 … researcher requires mutual-TLS client authentication but mutual-TLS is disabled on this node` (node retries) | Researcher has mTLS on, node has it off | Enable `[mtls]` on the node, register the researcher certificate, have the researcher register the node's |
| `FB628 … Declared node id … does not match the identity … its certificate is registered under` (researcher error) | A node declared an id different from the one its certificate is registered under | Investigate; the node id and its registered party id must be the same party |
| `FB628 … Refusing the node declaring id … its certificate is not registered` (researcher error) | The node completed the handshake but its certificate is absent from the registry — typically deleted while the researcher was running | Re-register the node's certificate, or leave it rejected if the removal was deliberate |
| `FB628 … Refusing the node declaring id … its certificate registry could not be read` (researcher error) | The certificate database could not be read even once, so no node can be identified. Look for the accompanying `certificate_store_unreadable` warning | Check the path and permissions of the `db` entry in the researcher config |
| `FB619 … no researcher certificate is registered` (node won't start) | mTLS on, researcher cert missing on node | Register the researcher certificate on the node |

!!! danger "A handshake failure may be an attack"
    Under mTLS a failed handshake is not silently retried as "server unavailable" — it
    is logged as a security event. If certificates are correct and a node still cannot
    connect, treat it as a possible man-in-the-middle and verify the endpoint.

## Development / testing shortcut

When several components run locally under the same Fed-BioMed installation, register in
each of them the certificates it must trust — the node certificates on the researcher,
the researcher's on each node. It expects one researcher and at least two nodes:

```shell
fedbiomed certificate-dev-setup
```

You still need to set `[mtls] enabled = True` in each config to enforce mutual TLS.
