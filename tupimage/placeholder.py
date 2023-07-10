import io
import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union, Callable

PLACEHOLDER_CHAR = "\U0010EEEE"

ROWCOLUMN_DIACRITICS = ['\U00000305', '\U0000030d', '\U0000030e', '\U00000310',
                        '\U00000312', '\U0000033d', '\U0000033e', '\U0000033f',
                        '\U00000346', '\U0000034a', '\U0000034b', '\U0000034c',
                        '\U00000350', '\U00000351', '\U00000352', '\U00000357',
                        '\U0000035b', '\U00000363', '\U00000364', '\U00000365',
                        '\U00000366', '\U00000367', '\U00000368', '\U00000369',
                        '\U0000036a', '\U0000036b', '\U0000036c', '\U0000036d',
                        '\U0000036e', '\U0000036f', '\U00000483', '\U00000484',
                        '\U00000485', '\U00000486', '\U00000487', '\U00000592',
                        '\U00000593', '\U00000594', '\U00000595', '\U00000597',
                        '\U00000598', '\U00000599', '\U0000059c', '\U0000059d',
                        '\U0000059e', '\U0000059f', '\U000005a0', '\U000005a1',
                        '\U000005a8', '\U000005a9', '\U000005ab', '\U000005ac',
                        '\U000005af', '\U000005c4', '\U00000610', '\U00000611',
                        '\U00000612', '\U00000613', '\U00000614', '\U00000615',
                        '\U00000616', '\U00000617', '\U00000657', '\U00000658',
                        '\U00000659', '\U0000065a', '\U0000065b', '\U0000065d',
                        '\U0000065e', '\U000006d6', '\U000006d7', '\U000006d8',
                        '\U000006d9', '\U000006da', '\U000006db', '\U000006dc',
                        '\U000006df', '\U000006e0', '\U000006e1', '\U000006e2',
                        '\U000006e4', '\U000006e7', '\U000006e8', '\U000006eb',
                        '\U000006ec', '\U00000730', '\U00000732', '\U00000733',
                        '\U00000735', '\U00000736', '\U0000073a', '\U0000073d',
                        '\U0000073f', '\U00000740', '\U00000741', '\U00000743',
                        '\U00000745', '\U00000747', '\U00000749', '\U0000074a',
                        '\U000007eb', '\U000007ec', '\U000007ed', '\U000007ee',
                        '\U000007ef', '\U000007f0', '\U000007f1', '\U000007f3',
                        '\U00000816', '\U00000817', '\U00000818', '\U00000819',
                        '\U0000081b', '\U0000081c', '\U0000081d', '\U0000081e',
                        '\U0000081f', '\U00000820', '\U00000821', '\U00000822',
                        '\U00000823', '\U00000825', '\U00000826', '\U00000827',
                        '\U00000829', '\U0000082a', '\U0000082b', '\U0000082c',
                        '\U0000082d', '\U00000951', '\U00000953', '\U00000954',
                        '\U00000f82', '\U00000f83', '\U00000f86', '\U00000f87',
                        '\U0000135d', '\U0000135e', '\U0000135f', '\U000017dd',
                        '\U0000193a', '\U00001a17', '\U00001a75', '\U00001a76',
                        '\U00001a77', '\U00001a78', '\U00001a79', '\U00001a7a',
                        '\U00001a7b', '\U00001a7c', '\U00001b6b', '\U00001b6d',
                        '\U00001b6e', '\U00001b6f', '\U00001b70', '\U00001b71',
                        '\U00001b72', '\U00001b73', '\U00001cd0', '\U00001cd1',
                        '\U00001cd2', '\U00001cda', '\U00001cdb', '\U00001ce0',
                        '\U00001dc0', '\U00001dc1', '\U00001dc3', '\U00001dc4',
                        '\U00001dc5', '\U00001dc6', '\U00001dc7', '\U00001dc8',
                        '\U00001dc9', '\U00001dcb', '\U00001dcc', '\U00001dd1',
                        '\U00001dd2', '\U00001dd3', '\U00001dd4', '\U00001dd5',
                        '\U00001dd6', '\U00001dd7', '\U00001dd8', '\U00001dd9',
                        '\U00001dda', '\U00001ddb', '\U00001ddc', '\U00001ddd',
                        '\U00001dde', '\U00001ddf', '\U00001de0', '\U00001de1',
                        '\U00001de2', '\U00001de3', '\U00001de4', '\U00001de5',
                        '\U00001de6', '\U00001dfe', '\U000020d0', '\U000020d1',
                        '\U000020d4', '\U000020d5', '\U000020d6', '\U000020d7',
                        '\U000020db', '\U000020dc', '\U000020e1', '\U000020e7',
                        '\U000020e9', '\U000020f0', '\U00002cef', '\U00002cf0',
                        '\U00002cf1', '\U00002de0', '\U00002de1', '\U00002de2',
                        '\U00002de3', '\U00002de4', '\U00002de5', '\U00002de6',
                        '\U00002de7', '\U00002de8', '\U00002de9', '\U00002dea',
                        '\U00002deb', '\U00002dec', '\U00002ded', '\U00002dee',
                        '\U00002def', '\U00002df0', '\U00002df1', '\U00002df2',
                        '\U00002df3', '\U00002df4', '\U00002df5', '\U00002df6',
                        '\U00002df7', '\U00002df8', '\U00002df9', '\U00002dfa',
                        '\U00002dfb', '\U00002dfc', '\U00002dfd', '\U00002dfe',
                        '\U00002dff', '\U0000a66f', '\U0000a67c', '\U0000a67d',
                        '\U0000a6f0', '\U0000a6f1', '\U0000a8e0', '\U0000a8e1',
                        '\U0000a8e2', '\U0000a8e3', '\U0000a8e4', '\U0000a8e5',
                        '\U0000a8e6', '\U0000a8e7', '\U0000a8e8', '\U0000a8e9',
                        '\U0000a8ea', '\U0000a8eb', '\U0000a8ec', '\U0000a8ed',
                        '\U0000a8ee', '\U0000a8ef', '\U0000a8f0', '\U0000a8f1',
                        '\U0000aab0', '\U0000aab2', '\U0000aab3', '\U0000aab7',
                        '\U0000aab8', '\U0000aabe', '\U0000aabf', '\U0000aac1',
                        '\U0000fe20', '\U0000fe21', '\U0000fe22', '\U0000fe23',
                        '\U0000fe24', '\U0000fe25', '\U0000fe26', '\U00010a0f',
                        '\U00010a38', '\U0001d185', '\U0001d186', '\U0001d187',
                        '\U0001d188', '\U0001d189', '\U0001d1aa', '\U0001d1ab',
                        '\U0001d1ac', '\U0001d1ad', '\U0001d242', '\U0001d243',
                        '\U0001d244']  # noqa

@dataclass(frozen=True)
class Rect:
    start_column: int = 0
    start_row: int = 0
    end_column: int = 0
    end_row: int = 0

    @property
    def width(self):
        return self.end_column - self.start_column

    @property
    def height(self):
        return self.end_row - self.start_row

    def clone_with(self, **kwargs):
        return dataclasses.replace(self, **kwargs)

class DiacriticLevel(Enum):
    NONE = 0
    ROW = 1
    ROW_COLUMN = 2
    ROW_COLUMN_ID4THBYTE = 3
    ROW_COLUMN_ID4THBYTE_IF_NONZERO = 4

@dataclass(frozen=True)
class ImagePlaceholderMode:
    allow_256colors_for_image_id: bool = True
    allow_256colors_for_placement_id: bool = False
    skip_placement_id_if_zero: bool = True
    first_column_diacritic_level: DiacriticLevel = DiacriticLevel.ROW_COLUMN_ID4THBYTE_IF_NONZERO
    other_columns_diacritic_level: DiacriticLevel = DiacriticLevel.ROW_COLUMN_ID4THBYTE_IF_NONZERO
    placeholder_char: str = PLACEHOLDER_CHAR

    def __post_init__(self):
        if firstcol_diacritic_count == DiacriticLevel.NONE:
            raise ValueError("first_column_diacritic_level cannot be NONE, at least the row must be printed")

    def clone_with(self, **kwargs):
        return dataclasses.replace(self, **kwargs)

    def only24bitcolors(self):
        return self.clone_with(allow_256colors_for_image_id=False, allow_256colors_for_placement_id=False)

    @staticmethod
    def standard():
        return ImagePlaceholderMode()

    @staticmethod
    def complete():
        return ImagePlaceholderMode(first_column_diacritic_level=DiacriticLevel.ROW_COLUMN_ID4THBYTE, other_columns_diacritic_level=DiacriticLevel.ROW_COLUMN_ID4THBYTE)

    @staticmethod
    def minimal():
        return ImagePlaceholderMode(first_column_diacritic_level=DiacriticLevel.ROW, other_columns_diacritic_level=DiacriticLevel.NONE)

@dataclass(frozen=True)
class ImagePlaceholder:
    image_id: int
    placement_id: int = 0
    start_column: int = 0
    start_row: int = 0
    end_column: int = 0
    end_row: int = 0

    def __post_init__(self):
        if self.image_id < 0 or self.image_id > 0xFFFFFFFF:
            raise ValueError("Image ID must be a 32-bit unsigned integer, but it is {self.image_id}")
        if self.placement_id is not None and (self.placement_id < 0 or self.placement_id > 0xFFFFFF):
            raise ValueError("Placement ID must be a 24-bit unsigned integer, but it is {self.placement_id}")
        if start_column < 0:
            raise ValueError("Start column must be non-negative, but it is {self.start_column}")
        if start_row < 0:
            raise ValueError("Start row must be non-negative, but it is {self.start_row}")
        if start_column >= end_column:
            raise ValueError("Start column must be less than end column, but {self.start_column} >= {self.end_column}")
        if start_row >= end_row:
            raise ValueError("Start row must be less than end row, but {self.start_row} >= {self.end_row}")
        if end_row > len(ROWCOLUMN_DIACRITICS):
            raise ValueError("End row must not be greater than {len(ROWCOLUMN_DIACRITICS)}, but it is {self.end_row}")

    def clone_with(self, **kwargs):
        return dataclasses.replace(self, **kwargs)

    def to_stream(self, stream: io.TextIOBase, mode: ImagePlaceholderMode=ImagePlaceholderMode.standard(), place: Optional[Rect]=None, no_escape: bool=False, additional_formatting: Union[str, Callable[[int], str]]=""):
        line_prefix = ""
        line_suffix = ""
        # Encode first 24 bits of IDs in the fg and underline colors.
        if not no_escape:
            line_prefix += "\033[0m"
            line_suffix += "\033[0m"
            if mode.allow_256colors_for_image_id and (self.image_id & 0xFFFF00 == 0):
                line_prefix += f"\033[38;5;{self.image_id & 0xFF}m"
            else:
                line_prefix += f"\033[38;2;{(self.image_id >> 16) & 0xFF};{(self.image_id >> 8) & 0xFF};{self.image_id & 0xFF}m"
            if mode.skip_placement_id_if_zero and self.placement_id == 0:
                if mode.allow_256colors_for_placement_id and (self.placement_id & 0xFFFF00 == 0):
                    line_prefix += f"\033[58;5;{self.placement_id & 0xFF}m"
                else:
                    line_prefix += f"\033[58;2;{(self.placement_id >> 16) & 0xFF};{(self.placement_id >> 8) & 0xFF};{self.placement_id & 0xFF}m"

        # Figure out how many diacritics to print.
        image_id_4thbyte = (self.image_id & 0xFF000000) >> 24
        image_id_4thbyte_diacritic = ROWCOLUMN_DIACRITICS[image_id_4thbyte]
        firstcol_diacritic_count = mode.first_column_diacritic_level.value
        othercol_diacritic_count = mode.other_columns_diacritic_level.value
        if image_id_4thbyte != 0:
            firstcol_diacritic_count = 3
            if mode.other_columns_diacritic_level == DiacriticLevel.ROW_COLUMN_ID4THBYTE_IF_NONZERO:
                other_diacritic_count = 3
        else:
            if mode.first_column_diacritic_level == DiacriticLevel.ROW_COLUMN_ID4THBYTE_IF_NONZERO:
                firstcol_diacritic_count = 2
            if mode.other_columns_diacritic_level == DiacriticLevel.ROW_COLUMN_ID4THBYTE_IF_NONZERO:
                other_diacritic_count = 2

        # Adjust the end row/column to fit the rectangle.
        end_row = self.end_row
        end_column = self.end_column
        if place is not None:
            end_row = min(end_row, self.start_row + place.height)
            end_column = min(end_column, self.start_column + place.width)

        # Print the placeholder.
        for row in range(self.start_row, end_row):
            
            if place:
                
            # Print line formatting: colors for IDs and user-specified
            # additional formatting.
            stream.write(line_prefix)
            if isinstance(additional_formatting, str):
                stream.write(additional_formatting)
            else:
                stream.write(additional_formatting(row))
            # The row diacritic.
            row_diacritic = ROWCOLUMN_DIACRITICS[row]
            # Print the placeholder and diacritics for the first column.
            stream.write(mode.placeholder_char)
            if firstcol_diacritic_count >= 1:
                stream.write(row_diacritic)
                if firstcol_diacritic_count >= 2:
                    stream.write(ROWCOLUMN_DIACRITICS[0])
                    if firstcol_diacritic_count >= 3:
                        stream.write(image_id_4thbyte_diacritic)
            # Print the placeholders with diacritics for other columns.
            for col in range(self.start_column + 1, end_column):
                stream.write(mode.placeholder_char)
                if firstcol_diacritic_count >= 1:
                    stream.write(row_diacritic)
                    if firstcol_diacritic_count >= 2 and col < len(ROWCOLUMN_DIACRITICS):
                        stream.write(ROWCOLUMN_DIACRITICS[col])
                        if firstcol_diacritic_count >= 3:
                            stream.write(image_id_4thbyte_diacritic)
            stream.write(line_suffix)
            if place is None:
                stream.write("\n")
