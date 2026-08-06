use crate::OcrBoundingBox;

// baidu/Unlimited-OCR's reviewed postprocessor divides generated coordinates
// by 999 before scaling them to the decoded image dimensions.
const GROUNDING_BASIS: u16 = 999;
const MAX_COORDINATE_BOXES_PER_MARKER: usize = 128;

pub(super) struct ParsedCoordinates {
    pub bounds: NormalizedBox,
    pub boxes: Vec<NormalizedBox>,
}

pub(super) fn parse_coordinates(raw: &str) -> Result<ParsedCoordinates, CoordinateParseError> {
    CoordinateParser::new(raw).parse()
}

pub(super) enum CoordinateParseError {
    Invalid,
    Limit,
}

struct CoordinateParser<'a> {
    bytes: &'a [u8],
    position: usize,
    boxes: Vec<NormalizedBox>,
    bounds: Option<NormalizedBox>,
}

impl<'a> CoordinateParser<'a> {
    fn new(raw: &'a str) -> Self {
        Self {
            bytes: raw.as_bytes(),
            position: 0,
            boxes: Vec::new(),
            bounds: None,
        }
    }

    fn parse(mut self) -> Result<ParsedCoordinates, CoordinateParseError> {
        self.skip_whitespace();
        self.expect(b'[')?;
        self.skip_whitespace();
        if self.peek() == Some(b'[') {
            loop {
                self.parse_box()?;
                self.skip_whitespace();
                match self.peek() {
                    Some(b',') => {
                        self.position += 1;
                    }
                    Some(b']') => {
                        self.position += 1;
                        break;
                    }
                    _ => return Err(CoordinateParseError::Invalid),
                }
            }
        } else {
            self.parse_box_body()?;
        }
        self.skip_whitespace();
        if self.position != self.bytes.len() {
            return Err(CoordinateParseError::Invalid);
        }
        let bounds = self.bounds.ok_or(CoordinateParseError::Invalid)?;
        Ok(ParsedCoordinates {
            bounds,
            boxes: self.boxes,
        })
    }

    fn parse_box(&mut self) -> Result<(), CoordinateParseError> {
        self.expect(b'[')?;
        self.parse_box_body()
    }

    fn parse_box_body(&mut self) -> Result<(), CoordinateParseError> {
        let left = self.coordinate()?;
        self.expect_separator()?;
        let top = self.coordinate()?;
        self.expect_separator()?;
        let right = self.coordinate()?;
        self.expect_separator()?;
        let bottom = self.coordinate()?;
        self.expect(b']')?;
        if left >= right || top >= bottom {
            return Err(CoordinateParseError::Invalid);
        }
        if self.boxes.len() >= MAX_COORDINATE_BOXES_PER_MARKER {
            return Err(CoordinateParseError::Limit);
        }
        let current = NormalizedBox {
            left,
            top,
            right,
            bottom,
        };
        self.boxes.push(current);
        self.bounds = Some(match self.bounds {
            Some(bounds) => bounds.union(current),
            None => current,
        });
        Ok(())
    }

    fn coordinate(&mut self) -> Result<u16, CoordinateParseError> {
        self.skip_whitespace();
        let start = self.position;
        let mut value = 0_u16;
        while let Some(byte) = self.peek().filter(u8::is_ascii_digit) {
            value = value
                .checked_mul(10)
                .and_then(|value| value.checked_add(u16::from(byte - b'0')))
                .ok_or(CoordinateParseError::Invalid)?;
            self.position += 1;
        }
        if self.position == start || value > GROUNDING_BASIS {
            return Err(CoordinateParseError::Invalid);
        }
        Ok(value)
    }

    fn expect_separator(&mut self) -> Result<(), CoordinateParseError> {
        self.expect(b',')
    }

    fn expect(&mut self, expected: u8) -> Result<(), CoordinateParseError> {
        self.skip_whitespace();
        if self.peek() != Some(expected) {
            return Err(CoordinateParseError::Invalid);
        }
        self.position += 1;
        Ok(())
    }

    fn skip_whitespace(&mut self) {
        while self.peek().is_some_and(|byte| byte.is_ascii_whitespace()) {
            self.position += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.position).copied()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct NormalizedBox {
    left: u16,
    top: u16,
    right: u16,
    bottom: u16,
}

impl NormalizedBox {
    fn union(self, other: Self) -> Self {
        Self {
            left: self.left.min(other.left),
            top: self.top.min(other.top),
            right: self.right.max(other.right),
            bottom: self.bottom.max(other.bottom),
        }
    }

    pub(super) fn to_source_pixels(
        self,
        image_width: u32,
        image_height: u32,
    ) -> Option<OcrBoundingBox> {
        let left = scale_coordinate(self.left, image_width);
        let top = scale_coordinate(self.top, image_height);
        let right = scale_coordinate(self.right, image_width);
        let bottom = scale_coordinate(self.bottom, image_height);
        Some(OcrBoundingBox {
            x: left,
            y: top,
            width: right.checked_sub(left).filter(|width| *width > 0)?,
            height: bottom.checked_sub(top).filter(|height| *height > 0)?,
        })
    }
}

fn scale_coordinate(value: u16, dimension: u32) -> u32 {
    ((u64::from(value) * u64::from(dimension)) / u64::from(GROUNDING_BASIS)) as u32
}
