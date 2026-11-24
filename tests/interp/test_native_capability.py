"""Native handshakes follow the current interpretation service lifetime."""
import pytest

from caliper import _controls, _wire
from caliper.interp._service import register_service, unregister_service


def test_native_handshake_tracks_service_registration_and_removal(monkeypatch):
    pytest.importorskip('caliper._netio')
    from websockets.sync.client import connect
    from caliper._net_native import NativeServer
    owner = _controls.Controls()
    monkeypatch.setattr(_controls, 'CONTROLS', owner)
    server = NativeServer(port=0)
    services = []

    def announced():
        with connect(server.stream_url, open_timeout=5, close_timeout=1, compression=None) as client:
            client.send(_wire.encode(_wire.Hello('viewer', [_wire.CAP_CONTROLS, _wire.CAP_INTERPRETATION],
                                                  {'name': 'fixture', 'version': '0'})))
            reply = _wire.decode(client.recv(timeout=5))
            assert isinstance(reply, _wire.Welcome)
            return _wire.CAP_INTERPRETATION in reply.capabilities

    try:
        assert not announced()
        first = register_service(controls=owner)
        services.append(first)
        assert announced()
        second = register_service(controls=owner)
        services.append(second)
        assert first.closed and announced()
        unregister_service(first, controls=owner)
        assert announced(), 'stale owner removal must retain the current service'
        other = _controls.Controls()
        unrelated = register_service(controls=other)
        try:
            unregister_service(unrelated, controls=other)
            assert announced(), 'a private Controls owner must not alter the native capability'
        finally:
            unrelated.close()
        unregister_service(second, controls=owner)
        assert not announced()
    finally:
        for service in services:
            if not service.closed:
                unregister_service(service, controls=owner)
        server.close()
