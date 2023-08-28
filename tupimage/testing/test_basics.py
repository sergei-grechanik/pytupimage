import tupimage
from tupimage import GraphicsTerminal, PutCommand, TransmitCommand
from tupimage.testing import TestingContext, screenshot_test


@screenshot_test
def test_text_printing(ctx: TestingContext):
    for i in range(24):
        ctx.write(str(i) + "\n")
    ctx.write(
        "This is a text-only test. If it fails, it means there is something"
        " wrong with how the terminal is configured (size, colors) or with the"
        " screenshot comparison algorithm.\n"
    )
    for i in range(33, 127):
        ctx.write(chr(i))
    ctx.take_screenshot(
        "Some text and all printable ascii characters. The text should be at"
        " the bottom of the screen."
    )


@screenshot_test
def test_text_colors(ctx: TestingContext):
    for i in range(16):
        for j in range(16):
            ctx.write(f"\033[48;5;{i}m\033[38;5;{j}m Aa ")
        ctx.write("\n")
    ctx.take_screenshot("")


@screenshot_test(suffix="placeholder", params={"placeholder": True})
@screenshot_test
def test_display_movecursor(ctx: TestingContext, placeholder: bool = False):
    term = ctx.term.clone_with(force_placeholders=placeholder)
    cmd = TransmitCommand(
        image_id=1,
        medium=tupimage.TransmissionMedium.FILE,
        quiet=tupimage.Quietness.QUIET_UNLESS_ERROR,
        format=tupimage.Format.PNG,
    )
    term.send_command(
        cmd.clone_with(image_id=1)
        .set_filename(ctx.get_wikipedia_png())
        .set_placement(rows=10, columns=20)
    )
    ctx.take_screenshot("Wikipedia logo. May be slightly stretched on kitty.")
    term.move_cursor(up=9)
    term.send_command(
        cmd.clone_with(image_id=2)
        .set_filename(ctx.get_column_png())
        .set_placement(rows=10, columns=5)
    )
    term.move_cursor(up=9)
    term.send_command(PutCommand(image_id=1, rows=10, columns=20, quiet=1))
    term.move_cursor(up=9)
    term.send_command(PutCommand(image_id=1, rows=5, columns=10, quiet=1))
    term.move_cursor(left=10, down=1)
    term.send_command(PutCommand(image_id=1, rows=5, columns=10, quiet=1))
    ctx.take_screenshot("Wikipedia logo and some columns.")


@screenshot_test(suffix="placeholder", params={"placeholder": True})
@screenshot_test
def test_display_nomovecursor(ctx: TestingContext, placeholder: bool = False):
    term = ctx.term.clone_with(force_placeholders=placeholder)
    cmd = TransmitCommand(
        image_id=1,
        medium=tupimage.TransmissionMedium.FILE,
        quiet=tupimage.Quietness.QUIET_UNLESS_ERROR,
        format=tupimage.Format.PNG,
    )
    term.send_command(
        cmd.clone_with(image_id=1)
        .set_filename(ctx.get_wikipedia_png())
        .set_placement(rows=10, columns=20, do_not_move_cursor=True)
    )
    ctx.take_screenshot(
        "Wikipedia logo (slightly stretched on kitty). The cursor should be at"
        " the top left corner."
    )
    term.move_cursor(right=20)
    term.send_command(
        cmd.clone_with(image_id=2)
        .set_filename(ctx.get_column_png())
        .set_placement(rows=10, columns=5, do_not_move_cursor=True)
    )
    term.move_cursor(right=5)
    term.send_command(
        PutCommand(
            image_id=1, rows=10, columns=20, quiet=1, do_not_move_cursor=True
        )
    )
    term.move_cursor(right=20)
    term.send_command(
        PutCommand(
            image_id=1, rows=5, columns=10, quiet=1, do_not_move_cursor=True
        )
    )
    term.move_cursor(down=5)
    term.send_command(
        PutCommand(
            image_id=1, rows=5, columns=10, quiet=1, do_not_move_cursor=True
        )
    )
    ctx.take_screenshot(
        "Wikipedia logo and some columns. The cursor should be at the top left"
        " corner of the last column image."
    )


@screenshot_test(suffix="placeholder", params={"placeholder": True})
@screenshot_test
def test_multisize(ctx: TestingContext, placeholder: bool = False):
    term = ctx.term.clone_with(force_placeholders=placeholder)
    cmd = TransmitCommand(
        medium=tupimage.TransmissionMedium.FILE,
        quiet=tupimage.Quietness.QUIET_UNLESS_ERROR,
        format=tupimage.Format.PNG,
    )
    term.send_command(
        cmd.clone_with(image_id=1).set_filename(ctx.get_tux_png())
    )
    for r in range(1, 5):
        start_col = 0
        for c in range(1, 10):
            term.move_cursor_abs(col=start_col)
            term.send_command(
                PutCommand(
                    image_id=1,
                    rows=r,
                    columns=c,
                    quiet=1,
                    do_not_move_cursor=True,
                )
            )
            start_col += c
        term.move_cursor_abs(col=0)
        term.move_cursor(down=r)
    ctx.take_screenshot(
        "A grid of penguins of various sizes. On kitty they may be stretched."
    )


@screenshot_test(suffix="placeholder", params={"placeholder": True})
@screenshot_test
def test_oob(ctx: TestingContext, placeholder: bool = False):
    term = ctx.term.clone_with(force_placeholders=placeholder)
    cmd = TransmitCommand(
        medium=tupimage.TransmissionMedium.FILE,
        quiet=tupimage.Quietness.QUIET_UNLESS_ERROR,
        format=tupimage.Format.PNG,
    )
    term.send_command(
        cmd.clone_with(image_id=1).set_filename(ctx.get_ruler_png())
    )
    for r in range(24):
        term.move_cursor_abs(row=r, col=80 - (24 - r))
        term.send_command(
            PutCommand(
                image_id=1, rows=1, columns=24, quiet=1, do_not_move_cursor=True
            )
        )
    term.move_cursor_abs(row=0, col=0)
    ctx.take_screenshot("A ruler that goes off the screen. Not to scale.")


@screenshot_test(suffix="placeholder", params={"placeholder": True})
@screenshot_test
def test_oob_down(ctx: TestingContext, placeholder: bool = False):
    term = ctx.term.clone_with(force_placeholders=placeholder)
    cmd = TransmitCommand(
        medium=tupimage.TransmissionMedium.FILE,
        quiet=tupimage.Quietness.QUIET_UNLESS_ERROR,
        format=tupimage.Format.PNG,
    )
    term.send_command(
        cmd.clone_with(image_id=1).set_filename(ctx.get_tux_png())
    )
    for r in range(3):
        term.send_command(
            PutCommand(
                image_id=1,
                rows=10,
                columns=20,
                quiet=1,
                do_not_move_cursor=False,
            )
        )
    ctx.take_screenshot(
        "Three penguins, arranged diagonally. The top one is cut off."
    )


@screenshot_test(suffix="placeholder", params={"placeholder": True})
@screenshot_test
def test_oob_down_nomovecursor(ctx: TestingContext, placeholder: bool = False):
    term = ctx.term.clone_with(force_placeholders=placeholder)
    cmd = TransmitCommand(
        medium=tupimage.TransmissionMedium.FILE,
        quiet=tupimage.Quietness.QUIET_UNLESS_ERROR,
        format=tupimage.Format.PNG,
    )
    term.send_command(
        cmd.clone_with(image_id=1).set_filename(ctx.get_tux_png())
    )
    for r in range(3):
        term.send_command(
            PutCommand(
                image_id=1,
                rows=10,
                columns=20,
                quiet=1,
                do_not_move_cursor=True,
            )
        )
        term.move_cursor(down=10)
    ctx.take_screenshot(
        "Three penguins arranged vertically. The bottom one is cut off because"
        " the terminal shouldn't introduce new lines when C=1."
    )
