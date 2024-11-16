#!/Users/andreas/pyicloud/venv/bin/python3

from pyicloud import PyiCloudService, utils
from builtins import input
from pyicloud.exceptions import PyiCloudFailedLoginException
from pprint import pprint
import sys

username = "sunniva.josefsen@hotmail.com"
password = "ci8&G32Np%"

try:
    api = PyiCloudService(username, password)

    if api.requires_2fa:
        print("Two-factor authentication required.")
        code = input("Enter the code you received of one of your approved devices: ")
        result = api.validate_2fa_code(code)
        print("Code validation result: %s" % result)

        if not result:
            print("Failed to verify security code")
            sys.exit(1)

        if not api.is_trusted_session:
            print("Session is not trusted. Requesting trust...")
            result = api.trust_session()
            print("Session trust result %s" % result)

            if not result:
                print("Failed to request trust. You will likely be prompted for the code again in the coming weeks")
    elif api.requires_2sa:
        import click
        print("Two-step authentication required. Your trusted devices are:")

        devices = api.trusted_devices
        for i, device in enumerate(devices):
            print(
                "  %s: %s" % (i, device.get('deviceName',
                "SMS to %s" % device.get('phoneNumber')))
            )

        device = click.prompt('Which device would you like to use?', default=0)
        device = devices[device]
        if not api.send_verification_code(device):
            print("Failed to send verification code")
            sys.exit(1)

        code = click.prompt('Please enter validation code')
        if not api.validate_verification_code(device, code):
            print("Failed to verify verification code")
            sys.exit(1)
       
except PyiCloudFailedLoginException:
    # If they have a stored password; we just used it and
    # it did not work; let's delete it if there is one.
    if utils.password_exists_in_keyring(username):
        utils.delete_password_in_keyring(username)

        message = "Bad username or password for {username}".format(username=username)
        password = None

        failure_count += 1
        if failure_count >= 3:
            raise RuntimeError(message)

        print(message, file=sys.stderr)

api = PyiCloudService(username, password)

print(api.trusted_devices)
