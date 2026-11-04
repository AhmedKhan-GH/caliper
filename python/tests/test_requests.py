"""Request transport tests: no model or GPU required."""
import json
import os
import socket
import threading
import time
import uuid

import pytest
from caliper.requests import RequestServer, MAX_MESSAGE, encode


@pytest.fixture
def server():
    with RequestServer('test-' + uuid.uuid4().hex[:12], ['echo', 'wait']) as server:
        def handler(req, emit, cancelled):
            if req['op'] == 'wait':
                emit({'text': 'hello λ'})
                cancelled.wait(3)
                return {'cancelled': cancelled.is_set()}
            if req['op'] == 'fail':
                raise ValueError('bad request')
            if req['op'] == 'oversized':
                return {'text': 'a' * MAX_MESSAGE}
            if req['op'] == 'large':
                return {'text': 'a' * (1024 * 1024)}
            if req['op'] == 'not_json':
                return {'value': object()}
            if req['op'] == 'not_finite':
                return {'value': float('nan')}
            return {'text': req.get('text', '')}
        worker = threading.Thread(target=server.serve, args=(handler,))
        worker.start()
        yield server
        server.close()
        worker.join(2)
        assert not worker.is_alive()
    assert not os.path.exists(server.path)


def client(server):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(3)
    s.connect(server.path)
    return s


def receive(f):
    return json.loads(f.readline())


def test_fragmentation_unicode_and_errors(server):
    with client(server) as s, s.makefile('rb') as f:
        request = encode({'id': 1, 'op': 'echo', 'text': 'λ\n世界'})
        for b in request:
            s.sendall(bytes([b]))
        assert receive(f) == {'id': 1, 'done': True, 'result': {'text': 'λ\n世界'}}
        s.sendall(encode({'id': 2, 'op': 'hello'}) + encode({'id': 3, 'op': 'fail'}))
        assert receive(f)['result']['protocol'] == 1
        assert receive(f)['error'] == 'bad request'


def test_stop_and_busy(server):
    with client(server) as s, s.makefile('rb') as f:
        s.sendall(encode({'id': 1, 'op': 'wait'}))
        assert receive(f)['event']['text'] == 'hello λ'
        s.sendall(encode({'id': 2, 'op': 'echo'}))
        assert 'busy' in receive(f)['error']
        s.sendall(encode({'id': 3, 'op': 'stop'}))
        replies = {r['id']: r for r in [receive(f), receive(f)]}
        assert replies[1]['result']['cancelled']
        assert replies[3]['result']['stopping']
        s.sendall(encode({'id': 4, 'op': 'echo'}))
        assert receive(f)['id'] == 4


def test_disconnect_cancels_and_reconnect(server):
    with client(server) as s, s.makefile('rb') as f:
        s.sendall(encode({'id': 1, 'op': 'wait'}))
        receive(f)
    end = time.monotonic() + 3
    while server.busy and time.monotonic() < end:
        time.sleep(.01)
    assert not server.busy
    with client(server) as s, s.makefile('rb') as f:
        s.sendall(encode({'id': 2, 'op': 'echo'}))
        assert receive(f)['id'] == 2


@pytest.mark.parametrize('message', [
    b'[]\n', b'{broken}\n', b'{"op":"echo"}\n', b'x' * MAX_MESSAGE,
    b'{"id":true,"op":"echo"}\n', b'{"id":1,"op":"echo","value":NaN}\n',
    b'[' * 1500 + b']' * 1500 + b'\n', b'{"id":1,"op":"\xff"}\n',
])
def test_invalid_frame_disconnects(server, message):
    with client(server) as s:
        try:
            s.sendall(message)
            assert s.recv(1) == b''
        except (BrokenPipeError, ConnectionResetError):
            pass
    with client(server) as s, s.makefile('rb') as f:
        s.sendall(encode({'id': 1, 'op': 'hello'}))
        assert receive(f)['done']


def test_size_and_permissions(server):
    assert os.stat(server.path).st_mode & 0o777 == 0o600
    with pytest.raises(ValueError):
        encode({'text': 'a' * MAX_MESSAGE})
    with pytest.raises(OSError):
        RequestServer(os.path.basename(server.path)[:-5])


@pytest.mark.parametrize('operation', ['oversized', 'not_json', 'not_finite'])
def test_invalid_handler_result_returns_error_and_remains_usable(server, operation):
    with client(server) as s, s.makefile('rb') as f:
        s.sendall(encode({'id': 1, 'op': operation}))
        reply = receive(f)
        assert reply['id'] == 1 and reply['done'] and reply['error']
        s.sendall(encode({'id': 2, 'op': 'echo'}))
        assert receive(f)['id'] == 2


def test_large_reply_tolerates_controller_between_frames(server):
    with client(server) as s, s.makefile('rb') as f:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
        s.sendall(encode({'id': 1, 'op': 'large'}))
        time.sleep(.3)
        assert len(receive(f)['result']['text']) == 1024 * 1024


def test_send_timeout_closes_partial_frame():
    from caliper.requests import _Peer

    class PartialSocket:
        shut = False
        closed = False

        def sendall(self, data):
            self.partial = data[:10]
            raise socket.timeout('slow controller')

        def shutdown(self, how):
            self.shut = True

        def close(self):
            self.closed = True

    sock = PartialSocket()
    peer = _Peer(sock)
    with pytest.raises(socket.timeout):
        peer.send({'text': 'unfinished frame'})
    assert sock.shut and sock.closed and peer.cancel.is_set()


def test_close_cancels_current_handler(server):
    with client(server) as s, s.makefile('rb') as f:
        s.sendall(encode({'id': 1, 'op': 'wait'}))
        assert receive(f)['event']
        server.close()
    end = time.monotonic() + 2
    while server.busy and time.monotonic() < end:
        time.sleep(.01)
    assert not server.busy


def test_session_owns_request_server():
    from caliper.live import SessionTransport

    class Client:
        session = 'test-' + uuid.uuid4().hex[:12]
        session_file = ''
        closed = False

        def attached(self):
            return False

        def close(self):
            self.closed = True

    with SessionTransport(Client(), 'test', None, None) as session:
        server = session.request_server(['echo'])
        assert os.path.exists(server.path)
        with pytest.raises(RuntimeError, match='already'):
            session.request_server()
    assert server.closed.is_set()
    assert not os.path.exists(server.path)
    with pytest.raises(RuntimeError, match='closed'):
        session.request_server()
