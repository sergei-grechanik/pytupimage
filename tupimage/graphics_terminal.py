import base64
import copy
import fcntl
import io
import os
import random
import select
import struct
import termios
import time
from typing import BinaryIO, List, Optional, TextIO, Tuple, Union

from tupimage import (
    GraphicsCommand,
    GraphicsResponse,
    PlacementData,
    PutCommand,
    TransmitCommand,
)
from tupimage.placeholder import (
    AdditionalFormatting,
    ImagePlaceholder,
    ImagePlaceholderMode,
)


class TtySettingsGuard:
    def __init__(self, term: "GraphicsTerminal"):
        self.term = term

    def __enter__(self):
        self.term.push_tty_settings()

    def __exit__(self, type, value, traceback):
        self.term.pop_tty_settings()


class ShellScriptBinaryIOHelper(BinaryIO):
    def __init__(self, shellscript_out: TextIO):
        self.shellscript_out: TextIO = shellscript_out

    @staticmethod
    def _escape_bytes(data: bytes) -> str:
        escaped = []
        for byte in data:
            c = chr(byte)
            if 32 <= byte <= 126 and c not in "\\%'":
                # Printable ASCII char.
                escaped.append(chr(byte))
            elif c == "\n":
                escaped.append("\\n")
            elif c == "\\":
                escaped.append("\\\\")
            elif c == "%":
                escaped.append("%%")
            else:
                escaped.append("\\{:03o}".format(byte))
        return "".join(escaped)

    @staticmethod
    def _split_data_into_chunks(data: bytes):
        ORD_0 = ord("0")
        ORD_9 = ord("9")
        ORD_A = ord("A")
        ORD_Z = ord("Z")
        ORD_a = ord("a")
        ORD_z = ord("z")
        ORD_PLUS = ord("+")
        ORD_SLASH = ord("/")
        ORD_EQUAL = ord("=")

        chunks = []
        current_chunk = bytearray()
        is_base64_chunk = None

        for byte in data:
            is_base64 = (
                (ORD_0 <= byte <= ORD_9)
                or (ORD_A <= byte <= ORD_Z)
                or (ORD_a <= byte <= ORD_z)
                or byte in (ORD_PLUS, ORD_SLASH, ORD_EQUAL)
            )

            if is_base64_chunk is None:
                is_base64_chunk = is_base64

            if is_base64 != is_base64_chunk:
                if current_chunk:
                    chunks.append(bytes(current_chunk))
                    current_chunk = bytearray()
                is_base64_chunk = is_base64

            current_chunk.append(byte)

        if current_chunk:
            chunks.append(bytes(current_chunk))

        return chunks

    @staticmethod
    def _try_base64(data: bytes) -> Optional[str]:
        if len(data) > 172 or len(data) < 2:
            return None
        try:
            decoded = base64.b64decode(data, validate=True)
            escaped = ShellScriptBinaryIOHelper._escape_bytes(decoded)
            # Avoid too many special characters in the decoded string.
            if len(escaped) > len(decoded) * 1.05:
                return None
            return escaped
        except:
            return None

    @staticmethod
    def write_to_shellscript(shellscript_out: TextIO, data: bytes, comment: str = ""):
        # First split `data` into chunks such that each chunk is either a potential
        # base64-encoded string or definitely not.
        chunks = ShellScriptBinaryIOHelper._split_data_into_chunks(data)
        # The format string may look like "_Ga=t,t=f;%s".
        formatstring = ""
        # Params are substituted into the format string, they may look like
        # "$(printf '/path/to/file' | base64)".
        params = []
        for chunk in chunks:
            base64 = ShellScriptBinaryIOHelper._try_base64(chunk)
            if base64 is not None:
                formatstring += "%s"
                params.append(f"\"$(printf '{base64}' | base64)\"")
            else:
                formatstring += ShellScriptBinaryIOHelper._escape_bytes(chunk)
        # The final shell script command.
        command = f"printf '{formatstring}'"
        if params:
            command += f" {' '.join(params)}"
        # Print the comment inline if it fits.
        if len(comment) + len(command) + 3 <= 80:
            shellscript_out.write(f"{command} # {comment}\n")
            shellscript_out.flush()
            return
        # Otherwise, print the comment on a separate line.
        shellscript_out.write(f"# {comment}\n{command}\n")
        shellscript_out.flush()

    def write(self, data: Union[bytes, bytearray]) -> int:
        data = bytes(data)
        ShellScriptBinaryIOHelper.write_to_shellscript(self.shellscript_out, data)
        return len(data)


class GraphicsTerminal:
    def __init__(
        self,
        tty_filename: Optional[str] = None,
        tty_out: Optional[BinaryIO] = None,
        tty_in: Optional[BinaryIO] = None,
        max_command_size: Optional[int] = None,
        force_placeholders: bool = False,
        num_tmux_layers: int = 0,
        shellscript_out: Optional[TextIO] = None,
    ):
        if tty_filename is not None:
            if tty_out is not None or tty_in is not None:
                raise ValueError("Cannot specify both tty_filename and tty_out/tty_in")
        if tty_out is None and tty_in is None and tty_filename is None:
            tty_filename = "/dev/tty"
        if tty_filename is not None:
            fd = os.open(tty_filename, os.O_RDWR | os.O_NOCTTY)
            tty_out = os.fdopen(fd, "wb", buffering=0)
            tty_in = os.fdopen(fd, "rb", buffering=0)
        assert tty_out is not None
        self.tty_in: Optional[BinaryIO] = tty_in
        self.tty_out: BinaryIO = tty_out
        self.max_command_size: Optional[int] = max_command_size
        self.force_placeholders: bool = force_placeholders
        self.num_tmux_layers: int = num_tmux_layers
        self.shellscript_out: Optional[TextIO] = shellscript_out
        self.old_term_settings = []
        self.tracked_cursor_position: Optional[Tuple[int, int]] = None

    def clone_with(
        self,
        force_placeholders: Optional[bool] = None,
        num_tmux_layers: Optional[int] = None,
    ):
        res = copy.copy(self)
        res.old_term_settings = []
        if force_placeholders is not None:
            res.force_placeholders = force_placeholders
        if num_tmux_layers is not None:
            res.num_tmux_layers = num_tmux_layers
        return res

    def _write_to_shellscript(self, data: bytes, comment: str = ""):
        if self.shellscript_out is not None:
            ShellScriptBinaryIOHelper.write_to_shellscript(
                self.shellscript_out, data, comment
            )

    def _write(self, data: bytes, comment: str = ""):
        self.tty_out.write(data)
        self._write_to_shellscript(data, comment)

    def write(self, string: Union[str, bytes]):
        if isinstance(string, str):
            string = string.encode("utf-8")
        self._write(string)
        self.tracked_cursor_position = None

    def get_graphics_command_template(self) -> bytes:
        template = b"\033_G%b\033\\"
        for _ in range(self.num_tmux_layers):
            template = b"\033Ptmux;%b\033\\" % template.replace(b"\033", b"\033\033")
        return template

    def print_placeholder(
        self,
        placeholder: Optional[ImagePlaceholder] = None,
        image_id: Optional[int] = None,
        placement_id: Optional[int] = None,
        start_col: Optional[int] = None,
        start_row: Optional[int] = None,
        end_col: Optional[int] = None,
        end_row: Optional[int] = None,
        pos: Optional[Tuple[int, int]] = None,
        mode: ImagePlaceholderMode = ImagePlaceholderMode.default(),
        formatting: AdditionalFormatting = None,
        use_save_cursor: bool = True,
    ):
        if placeholder is None:
            placeholder = ImagePlaceholder()
        if image_id is not None:
            placeholder.image_id = image_id
        if placement_id is not None:
            placeholder.placement_id = placement_id
        if start_col is not None:
            placeholder.start_col = start_col
        if start_row is not None:
            placeholder.start_row = start_row
        if end_col is not None:
            placeholder.end_col = end_col
        if end_row is not None:
            placeholder.end_row = end_row
        placeholder.to_stream(
            self.tty_out,
            pos=pos,
            mode=mode,
            formatting=formatting,
            use_save_cursor=use_save_cursor,
        )
        if self.shellscript_out is not None:
            self.shellscript_out.write(f"# Placeholder {placeholder}\n")
            placeholder.to_stream(
                ShellScriptBinaryIOHelper(self.shellscript_out),
                pos=pos,
                mode=mode,
                formatting=formatting,
                use_save_cursor=use_save_cursor,
            )
            self.shellscript_out.write("\n")

    def print_placeholder_for_put(
        self,
        put_command: PutCommand,
        mode: ImagePlaceholderMode = ImagePlaceholderMode.default(),
    ):
        if put_command.rows is None or put_command.cols is None:
            raise ValueError(
                f"put_command must have known rows and cols: {put_command}"
            )
        if put_command.image_id is None:
            raise ValueError(f"put_command must have a known image_id: {put_command}")
        placement_id = put_command.placement_id
        if placement_id is None:
            placement_id = 0
        term_cols, term_rows = self.get_size()
        cur_x, cur_y = self.get_cursor_position_tracked()
        cols = min(put_command.cols, term_cols - cur_x)
        rows = put_command.rows
        if term_rows - cur_y < rows:
            if put_command.do_not_move_cursor:
                rows = term_rows - cur_y
            else:
                newlines = rows - (term_rows - cur_y)
                self._write(
                    b"\033[%dS" % newlines,
                    comment=f"Move cursor down by {newlines}",
                )
                self.move_cursor(up=newlines)
        if cols <= 0 or rows <= 0:
            return
        placeholder = ImagePlaceholder(
            image_id=put_command.image_id,
            placement_id=placement_id,
            end_col=cols,
            end_row=rows,
        )
        cur_x, cur_y = self.get_cursor_position()
        self.tracked_cursor_position = None
        self.print_placeholder(placeholder, mode=mode)
        if put_command.do_not_move_cursor:
            self.move_cursor_abs(col=cur_x, row=cur_y)
        elif cur_x + cols >= term_cols:
            # Newline
            self._write(b"\033E", comment="Move cursor to start of next line")
            self.set_tracked_cursor_position(0, cur_y + rows)
        else:
            self.set_tracked_cursor_position(cur_x + cols, cur_y + rows - 1)
        self.tty_out.flush()

    def send_command(
        self,
        command: GraphicsCommand,
        force_placeholders: Optional[bool] = None,
    ):
        # Convert a classic placement to a unicode placeholder placement if requested.
        if force_placeholders is None:
            force_placeholders = self.force_placeholders
        need_to_print_placeholder = False
        if force_placeholders:
            if (
                isinstance(command, TransmitCommand)
                and command.placement is not None
                and not command.placement.virtual
            ):
                need_to_print_placeholder = True
                command = command.clone_with()
                assert command.placement is not None
                command.placement.virtual = True
                if command.placement.placement_id is None:
                    command.placement.placement_id = random.randint(1, 2**24 - 1)
            if isinstance(command, PutCommand) and not command.virtual:
                need_to_print_placeholder = True
                command = command.clone_with(virtual=True)
                if command.placement_id is None:
                    command.placement_id = random.randint(1, 2**24 - 1)

        # If we want to log the commands to a shell script, we need a callback.
        callback = None
        if self.shellscript_out is not None:
            callback = lambda cmd: self._write_to_shellscript(cmd.to_bytes())
        # Send the command.
        command.send(
            self.tty_out,
            template=self.get_graphics_command_template(),
            max_size=self.max_command_size,
            callback=callback,
        )

        # Print a placeholder if needed.
        if need_to_print_placeholder:
            if isinstance(command, TransmitCommand):
                put_command = command.get_put_command()
                if put_command is not None:
                    self.print_placeholder_for_put(put_command)
            if isinstance(command, PutCommand):
                self.print_placeholder_for_put(command)

    def set_immediate_input_noecho(self):
        if self.tty_in is None:
            raise ValueError("Cannot set immediate input on a write-only terminal")
        settings = list(termios.tcgetattr(self.tty_in.fileno()))
        settings[3] &= ~(termios.ECHO | termios.ICANON)
        termios.tcsetattr(self.tty_in.fileno(), termios.TCSADRAIN, settings)

    def receive_response(self, timeout: float = 10) -> GraphicsResponse:
        if self.tty_in is None:
            raise ValueError("Cannot receive response on a write-only terminal")
        with self.guard_tty_settings():
            self.set_immediate_input_noecho()
            buffer = b""
            is_graphics_response = False
            end_time = time.time() + timeout
            while True:
                ready, _, _ = select.select([self.tty_in], [], [], timeout)
                if ready:
                    buffer += self.tty_in.read(1)
                    if is_graphics_response:
                        if buffer.endswith(b"\033\\"):
                            break
                    else:
                        if buffer.endswith(b"\033_G"):
                            is_graphics_response = True
                timeout = end_time - time.time()
                if timeout < 0:
                    return GraphicsResponse(is_valid=False, non_response=buffer)
            # Now parse the response
            res = GraphicsResponse(is_valid=True)
            non_response, response = buffer.split(b"\033_G", 2)
            res.non_response = non_response
            resp_and_message = response[:-2].split(b";", 2)
            if len(resp_and_message) > 1:
                res.message = resp_and_message[1].decode("utf-8")
                res.is_ok = resp_and_message[1] == b"OK"
            for part in resp_and_message[0].split(b","):
                try:
                    if part.startswith(b"i="):
                        res.image_id = int(part[2:])
                    elif part.startswith(b"I="):
                        res.image_number = int(part[2:])
                    elif part.startswith(b"p="):
                        res.placement_id = int(part[2:])
                    else:
                        key_and_val = part.split(b"=", 2)
                        res.additional_data[key_and_val[0].decode("utf-8")] = (
                            key_and_val[1].decode("utf-8")
                            if len(key_and_val) > 1
                            else None
                        )
                except ValueError:
                    pass
            return res

    def receive_multiple_responses(
        self, timeout: float = 0.01
    ) -> List[GraphicsResponse]:
        res = []
        while True:
            r = self.receive_response(timeout=timeout)
            if not r.is_valid:
                break
            res.append(r)
        return res

    def get_cursor_position(self, timeout: float = 2.0) -> Tuple[int, int]:
        with self.guard_tty_settings():
            self.set_immediate_input_noecho()
            # Don't use self._write here since we don't want to record this in
            # the generated shell script.
            self.tty_out.write(b"\033[6n")
            buffer = b""
            end_time = time.time() + timeout
            is_response = False
            while True:
                ready, _, _ = select.select([self.tty_in], [], [], timeout)
                if ready:
                    buffer += self.tty_in.read(1)
                    if is_response:
                        if buffer.endswith(b"R"):
                            break
                    else:
                        if buffer.endswith(b"\033["):
                            is_response = True
                            buffer = b""
                timeout = end_time - time.time()
                if timeout < 0:
                    raise TimeoutError(
                        "No response to cursor position request: %r" % buffer
                    )
            # Now parse the response
            parts = buffer[:-1].split(b";")
            if len(parts) != 2:
                raise ValueError(
                    "Invalid response to cursor position request: %r" % buffer
                )
            y, x = parts
            self.tracked_cursor_position = (int(x) - 1, int(y) - 1)
        return self.tracked_cursor_position

    def get_cursor_position_tracked(self, timeout: float = 2.0) -> Tuple[int, int]:
        if self.tracked_cursor_position is None:
            self.get_cursor_position(timeout=timeout)
        return self.tracked_cursor_position

    def push_tty_settings(self):
        self.old_term_settings.append(termios.tcgetattr(self.tty_in.fileno()))

    def pop_tty_settings(self):
        termios.tcsetattr(
            self.tty_in.fileno(),
            termios.TCSADRAIN,
            self.old_term_settings.pop(),
        )

    def guard_tty_settings(self) -> TtySettingsGuard:
        return TtySettingsGuard(self)

    def wait_keypress(self) -> bytes:
        with self.guard_tty_settings():
            self.set_immediate_input_noecho()
            result = b""
            while len(result) < 256:
                result += self.tty_in.read(1)
                ready, _, _ = select.select([self.tty_in], [], [], 0)
                if not ready:
                    break
            return result

    def detect_tmux(self):
        term = os.environ.get("TERM", "")
        if os.environ.get("TMUX") and ("screen" in term or "tmux" in term):
            self.num_tmux_layers = max(1, self.num_tmux_layers)
        else:
            self.num_tmux_layers = 0

    def reset(self):
        self._write(b"\033c", comment="Reset terminal")
        self.tty_out.flush()
        self.tracked_cursor_position = (0, 0)

    def _get_sizes(self) -> Tuple[int, int, int, int]:
        try:
            fileno = (
                self.tty_in.fileno()
                if self.tty_in is not None
                else self.tty_out.fileno()
            )
            return struct.unpack(
                "HHHH",
                fcntl.ioctl(
                    fileno,
                    termios.TIOCGWINSZ,
                    struct.pack("HHHH", 0, 0, 0, 0),
                ),
            )
        except OSError:
            return 0, 0, 0, 0

    def get_size(self) -> Tuple[int, int]:
        lines, cols, _, _ = self._get_sizes()
        if lines == 0 or cols == 0:
            raise ValueError("Could not determine terminal size")
        return (cols, lines)

    def get_cell_size(self) -> Optional[Tuple[int, int]]:
        lines, cols, width, height = self._get_sizes()
        if lines == 0 or cols == 0 or width == 0 or height == 0:
            return None
        return (width // cols, height // lines)

    def set_tracked_cursor_position(
        self,
        x: int,
        y: int,
        *,
        columns: Optional[int] = None,
        lines: Optional[int] = None,
    ):
        if columns is None or lines is None:
            columns, lines = self.get_size()
        self.tracked_cursor_position = (
            min(x, columns - 1),
            min(y, lines - 1),
        )

    def move_cursor(
        self,
        *,
        right: Optional[int] = None,
        down: Optional[int] = None,
        left: Optional[int] = None,
        up: Optional[int] = None,
    ):
        if up is not None:
            if down is not None:
                raise ValueError("Cannot specify both up and down")
            down = -up
        if left is not None:
            if right is not None:
                raise ValueError("Cannot specify both left and right")
            right = -left
        if down:
            if down > 0:
                self._write(b"\033[%dB" % down, comment=f"Move cursor down by {down}")
            else:
                self._write(b"\033[%dA" % -down, comment=f"Move cursor up by {-down}")
        if right:
            if right > 0:
                self._write(
                    b"\033[%dC" % right, comment=f"Move cursor right by {right}"
                )
            else:
                self._write(
                    b"\033[%dD" % -right,
                    comment=f"Move cursor left by {-right}",
                )
        self.tty_out.flush()
        if self.tracked_cursor_position is not None:
            self.set_tracked_cursor_position(
                self.tracked_cursor_position[0] + (right or 0),
                self.tracked_cursor_position[1] + (down or 0),
            )

    def move_cursor_abs(
        self,
        *,
        col: Optional[int] = None,
        row: Optional[int] = None,
        pos: Optional[Tuple[int, int]] = None,
    ):
        if pos is not None:
            if row is not None or col is not None:
                raise ValueError("Cannot specify both pos and row/col")
            col, row = pos
        if row is not None:
            self._write(b"\033[%dd" % (row + 1), comment=f"Move cursor to row {row}")
        if col is not None:
            self._write(b"\033[%dG" % (col + 1), comment=f"Move cursor to column {col}")
        self.tty_out.flush()
        if self.tracked_cursor_position is not None:
            self.set_tracked_cursor_position(
                col or self.tracked_cursor_position[0],
                row or self.tracked_cursor_position[1],
            )
        elif row is not None and col is not None:
            self.set_tracked_cursor_position(
                col,
                row,
            )

    def set_margins(self, top: int, bottom: int):
        self._write(
            b"\033[%d;%dr" % (top + 1, bottom + 1),
            comment=f"Set margins to {top}-{bottom}",
        )
        self.tty_out.flush()
        self.tracked_cursor_position = None

    def scroll_down(self, lines: int = 1):
        self._write(b"\033[%dS" % lines, comment=f"Scroll down by {lines}")
        self.tty_out.flush()
        self.tracked_cursor_position = None

    def scroll_up(self, lines: int = 1):
        self._write(b"\033[%dT" % lines, comment=f"Scroll up by {lines}")
        self.tty_out.flush()
        self.tracked_cursor_position = None
