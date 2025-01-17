# Fed-BioMed VPN/containers software dependencies

 The following packages are required for Fed-BioMed with VPN/containers:

 * [`docker`](https://docs.docker.com)
 * [`docker compose` v2](https://docs.docker.com/compose): don't confuse it with the obsolete `docker-compose` v1

### Install `docker` and `docker compose`

#### Linux Fedora

Install and start [docker engine packages](https://docs.docker.com/engine/install/fedora/). In simple cases it is enough to run :

```
$ sudo dnf install -y dnf-plugins-core
$ sudo dnf config-manager \
    --add-repo \
    https://download.docker.com/linux/fedora/docker-ce.repo
$ sudo dnf install -y docker-ce docker-ce-cli containerd.io
$ sudo systemctl start docker
```

Allow current account to use docker :

```
$ sudo usermod -aG docker $USER
```

Check with the account used to run Fed-BioMed that docker is up and can be used by the current account without error :

```
$ docker run hello-world
```

Install `docker compose`:
```
$ sudo dnf install -y docker-compose-plugin
```

To make sure you have a docker compose v2, you can run the following:
```
$ docker compose version
```

#### MacOS

Install `docker` and `docker compose` choosing one of the available options for example :

* official full [Docker Desktop](https://docs.docker.com/desktop/mac/install/) installation process, please check product license
* your favorite third party package manager for example :
    * macports provides [docker](https://ports.macports.org/port/docker/) port
    * homebrew provides [docker](https://formulae.brew.sh/formula/docker) formula
    * don't use the `docker-compose` v1 from macports or homebrew !
    * for `docker compose` v2, adapt the [manual plugin install procedure](https://docs.docker.com/compose/install/linux/#install-the-plugin-manually) by picking the [proper binary for your hardware](https://github.com/docker/compose/releases)

Check with the account used to run Fed-BioMed docker is up and can be used by the current account without error :

```
$ docker run hello-world
```

#### Other

Connect under an account with administrator privileges, install [`docker`](https://docs.docker.com/engine/install), ensure it is started and give docker privilege for the account used for running Fed-BioMed. Also install [`docker compose` v2](https://docs.docker.com/compose/install/).

Check with the account used to run Fed-BioMed docker is up and can be used by the current account without error :

```
$ docker run hello-world
```
