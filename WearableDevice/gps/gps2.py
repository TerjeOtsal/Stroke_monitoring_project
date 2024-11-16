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

    if api.requires_2sa:
        print("\nTwo-step authentication required.", "\nYour trusted devices are:")

        devices = api.trusted_devices
        for i, device in enumerate(devices):
            print(
                "    %s: %s"
                % (i, device.get("deviceName", "SMS to %s" % device.get("phoneNumber")))
            )

            print("\nWhich device would you like to use?")
            device = int(input("(number) --> "))
            device = devices[device]
            if not api.send_verification_code(device):
                print("Failed to send verification code")
                sys.exit(1)

            print("\nPlease enter validation code")
            code = input("(string) --> ")
            if not api.validate_verification_code(device, code):
                print("Failed to verify verification code")
                sys.exit(1)

            print("")
            break
        print(api.trusted_devices)
       
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


