# Specific instructions for Windows installation

**Fed-BioMed requires Windows 11, WSL2 and docker. It can run on a physical machine or a virtual machine.**

This documentation gives the steps for a typical Windows 11 installation, steps may vary depending on your system.


## Step 0: (optional) virtual machine

Skip this step if running Windows on a physical (native) machine, not a virtual machine.

Requirement : choose an hypervisor compatible with other installation

* **VMware Workstation 15.5.5 or above** can be used, it is compatible with [HyperV](https://blogs.vmware.com/workstation/2020/05/vmware-workstation-now-supports-hyper-v-mode.html). We successfully ran Fed-BioMed with *VMware Workstation 16*.
* VirtualBox 6.1 cannot be used as [it conflicts](https://docs.microsoft.com/en-us/troubleshoot/windows-client/application-management/virtualization-apps-not-work-with-hyper-v) with Hyper-V "Virtual Machine Platform" which is needed for WSL2

Tips :

* **enable virtualization engine** for your VM. For VMware Workstation 16, check the boxes in *Virtual Machine Settings > Hardware > Processors > Virtualization Engine*
* **allocate 8GB RAM or more** to the VM, this is needed for conda PyTorch installation. For VMware Workstation 16 this can be done in *Virtual Machine Settings > Hardware > Memory*

## Step 1: Windows

Requirement : **tested under Windows 11** 21H2, though Windows 10 version 2004 and higher (Build 19041 and higher) also [support WSL2](https://docs.microsoft.com/en-us/windows/wsl/install) and Docker Desktop.

* **[update](https://support.microsoft.com/en-au/windows/update-windows-3c5ae7fc-9fb6-9af1-1984-b5e0412c556a) Windows**
* **reboot** Windows

Requirement: Windows Enterprise, Pro or Education edition (needed for Hyper-V functionality, which is not present in Home edition)

Requirement : Hyper-V "Virtual Machine Platform" activation

* **[enable](https://techcommunity.microsoft.com/t5/educator-developer-blog/step-by-step-enabling-hyper-v-for-use-on-windows-11/ba-p/3745905) Hyper-V**
* **reboot** Windows


## Step 2: WSL

[WSL](https://docs.microsoft.com/en-us/windows/wsl/install) (Windows Subsystem for Linux) is a tool that allows to run Linux within a Windows system.
Version 2 of WSL is needed for docker.
We successfully tested Fed-BioMed with **Ubuntu-22.04** distribution.

Requirement : WSL version 2

* **activate WSL** with *main menu > enter 'Turn Windows Feature on or off' > and click on 'Windows Subsystem for Linux' checkbox*
* **reboot** Windows
* **set version 2** in a Windows command tool
```
cmd> wsl --set-default-version 2
```
* **reboot** Windows

Requirement : a WSL distribution, eg Ubuntu

* **install a distribution** in a Windows command tool
```
cmd> wsl --install -d Ubuntu-22.04
```
* if required by install, update the WSL2 kernel: in *main menu > enter 'Windows Update' > select 'Advanced Options' > activate 'Receive updates for other Microsoft products'. Then in Windows Update, install the last version of WSL2.
* **reboot** Windows

Check that WSL uses version 2 and Ubuntu is installed in a Windows command tool :
```
cmd> wsl -l -v
  NAME                   STATE           VERSION
* Ubuntu                 Running         2
```

Open a WSL session from a Windows command tool :
```
cmd> wsl
user@wsl-ubuntu$
```


## Step 3: docker

Requirement : `docker` and `docker compose`

Install Docker Desktop on the Windows machine :

  - **install [Docker Desktop](https://hub.docker.com/editions/community/docker-ce-desktop-windows)** in Windows. Check the product license.
  - **reboot** Windows

Setup docker in your WSL Ubuntu :

  - open an administrator session in WSL Ubuntu :

    ```
    user@wsl-ubuntu$ sudo bash
    root@wsl-ubuntu#
    ```
  - install [docker engine](https://docs.docker.com/engine/install/ubuntu/) as admin (root) account in WSL Ubuntu. Please note that `docker container run hello-world` will not work until we complete the steps below
  - if you use an account named `USER` under Ubuntu, authorize it to use docker by typing under an admin (root) account in WSL Ubuntu :

    ```
    root@wsl-ubuntu# adduser USER docker
    ```

  - open a **new** WSL Ubuntu terminal so that it is authorized to use docker

Check that you can use docker with your user account under Ubuntu :

```
user@wsl-ubuntu$ docker container run hello-world
```


## Step 4: conda

Requirement : conda installed in Ubuntu and configured for your user account

* install [Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) under Ubuntu, using your user account
* during installation, answer *Yes* to question *“Do you wish the installer to initialize Anaconda3 by running conda init?”*
* activate conda for your Ubuntu session

    ```
    user@wsl-ubuntu$ source ~/.bashrc
    ```


## Step 5: Fed-BioMed

Follow Fed-BioMed Linux installation tutorial [from the *git clone* command](../../tutorials/installation/0-basic-software-installation.md#install-fedbiomed-software)

When running ```network``` for the first time, a Windows defender pop up may appear (triggered by docker), choose *"authorize only on private network"*.

!!! info "Performance issue"
    To ensure  Fed-BioMed performance in WSL2, be sure to use the native WSL2 Linux filesystem (in `/home/login`),
    not the Windows filesystem (in `/mnt/c/Users/login`), both for cloning the library and storing datasets.
    We experienced 10-50 time slower execution with native Windows filesystem.
    This point is [documented by Microsoft](https://learn.microsoft.com/en-us/windows/wsl/compare-versions#exceptions-for-using-wsl-1-rather-than-wsl-2)

    You may also need to [increase the memory available to WSL2](https://learn.microsoft.com/en-us/answers/questions/1296124/how-to-increase-memory-and-cpu-limits-for-wsl2-win).
    By default, only 50% of the host's RAM is made available.


## Troubleshooting

### Step 2 troubleshooting

#### Error 1: ```The virtual machine could not be started because a required feature is not installed```


When installing Linux, if this error happens:

```
Installing, this may take a few minutes...
WslRegisterDistribution failed with error: 0x80370102
Error: 0x80370102 The virtual machine could not be started because a required feature is not installed.
```

This means either you need to enable virtualisation on the bios of your computer or to enable Hyper-V : for the latter, go in

```
main menu > enter 'Turn Windows Feature on or off' > and click on Hyper-V checkbox

```

Then restart Linux distribution installation.

#### Error 2: ```failed with error: 0x80004005```

If this error happen when installing a Linux disribution:

```
WSLRegisterDistribution failed with error: 0x80004005
```

- Press ```Win + R```: it will open a window named ```Run tasks```
- Enter ```REGEDIT``` and hit ```OK```
- Navigate to : ```HKEY_LOCAL_MACHINE\CurrentControlSet\Services\LxssManager```
- Set the **value Data** to 2 and exit ```REGEDIT```.
- reboot the machine and see if it is working


### Step 5 troubleshooting:


If we encounter 'Operation not permitted' error on cloning git repository, you may follow below steps.

Error --> fatal: could not set 'core.filemode' to 'false'

1. Launch Ubuntu WSL.

2. Create the file /etc/wsl.conf if it doesn't exist.

3. Open the file (nano /etc/wsl.conf) and add the below lines:
    [automount]
    options = "metadata"
    
4. Save the file and shoutdown WSL

5. Relaunch Ubuntu WSL

If the problem still persists, you may try restarting the machine and then execute git clone command.

We detail here two common issues encountered when instlling and running `Fed-BioMed`.

- If launching Fed-BioMed researcher fails with a message mentioning a display error, you may need to use an alternate IP address. There are two ways of fixing this:

    1. run jupyter ```jupyter notebook --ip $(python3 -c "import subprocess; subprocess.run(['hostname', "-I"], text=True).sdtout")``` and connect to the IP address given by jupyter-notebook.
    2. or you can just connect to the IP address given by the command ```ip addr | grep eth0 | grep inet``` instead of connecting to ```localhost```.

- If Fed-BioMed fails with a message mentioning a `System.Management.Automation` error you may need to give WSL which browser to use. Set the `BROWSER` environment variable to the path to the browser. For example to use Microsoft Edge the path and command are commonly :
```
user@wsl-ubuntu$ export BROWSER=export BROWSER=/mnt/c/Program\ Files\ \(x86\)/Microsoft/Edge/Application/msedge.exe
```
