def scp_to_pc(source, target='NEWCOOL',
              hostname='kimjg.astro.princeton.edu', username='jgkim'):
    """Function to copy files to my directory
    """
    from paramiko import SSHClient
    from scp import SCPClient

    target = '~/Dropbox/Apps/Overleaf/{0:s}/figures'.format(target)

    try:
        client = SSHClient()
        client.load_system_host_keys()
        client.connect(hostname,username=username)
        with SCPClient(client.get_transport()) as scp:
            scp.put(source, target)
    finally:
        if client:
            client.close()
