const fs = require("fs");
const path = require("path");

const html = fs.readFileSync(path.join(__dirname, "index.html"), "utf8");
const required = ["No validated strategy", "Invalidated", "Negative", "Uncertain"];
const forbidden = ["Two validated", "ACCEPT-able for live deployment", "all 3 honest-PnL gates PASS"];

for (const text of required) {
  if (!html.includes(text)) {
    throw new Error("missing sober status text: " + text);
  }
}
for (const text of forbidden) {
  if (html.includes(text)) {
    throw new Error("stale deployment claim remains: " + text);
  }
}

console.log("dashboard smoke passed");
