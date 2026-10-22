# ACRS Demo shadow deployment

These units run the current Binance Spot Demo universe reconstruction and
signal shadow once per UTC day. They do not submit orders. The execution gate
is hard-coded closed in the current shadow pipeline, and the broker adapter
does not expose order or cancel methods.

Install on the designated server only after the local commit has been pushed
and pulled there:

```text
sudo cp deploy/systemd/acrs-demo-shadow.service /etc/systemd/system/
sudo cp deploy/systemd/acrs-demo-shadow.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now acrs-demo-shadow.timer
sudo systemctl start acrs-demo-shadow.service
```

Inspect the first run:

```text
sudo systemctl status acrs-demo-shadow.timer --no-pager
sudo systemctl status acrs-demo-shadow.service --no-pager
sudo journalctl -u acrs-demo-shadow.service -n 80 --no-pager
systemctl list-timers acrs-demo-shadow.timer --no-pager
```

The environment file remains outside the repository at
`/home/cryptobot/.config/acrs/binance_demo.env` with mode `600`.
