import { test, expect } from '@playwright/test';
import { registerInterpPieceTests } from '../../tests/interp_pieces/browser.js';
registerInterpPieceTests(test, expect);
