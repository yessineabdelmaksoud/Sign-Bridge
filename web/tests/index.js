// Entry point for `node --test web/tests/`, the command in web/CONTRACT.md.
// Node 22 does not look inside a directory given to --test: it runs the
// directory as a module, which resolves to this file. `cd web && node --test`
// finds the *.test.js files on its own and does not need it.
import "./export.test.js";
import "./landmarks.test.js";
import "./features.test.js";
import "./capture.test.js";
import "./model.test.js";
import "./recognizer.test.js";
import "./pace.test.js";
import "./channel.test.js";
import "./lexicon.test.js";
import "./speech.test.js";
import "./avatar.test.js";
import "./qr.test.js";
import "./invite.test.js";
