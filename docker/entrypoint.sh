#!/bin/sh

# Add user to /etc/passwd if it doesn't exist
if ! id -u $USER_ID >/dev/null 2>&1; then
    echo "myuser:x:$USER_ID:$GROUP_ID::/home/myuser:/bin/sh" >> /etc/passwd
    echo "myuser:x:$GROUP_ID:" >> /etc/group
    mkdir -p /home/myuser
    chown myuser:myuser -R /home/myuser
    echo "myuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
    usermod -U myuser  # Unlock the user account
     echo "myuser:password" | chpasswd  # Set a password for the user
fi

exec gosu myuser "$@"
