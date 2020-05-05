import JSONEditor from 'jsoneditor';
import 'jsoneditor/dist/jsoneditor.css';
import 'jsoneditor/dist/img/jsoneditor-icons.svg';

const defaultConfig = require('./config.json');
const container = document.getElementById("editor");
const editor = new JSONEditor(container, { mode: 'code' });
let shown = false;

let config = defaultConfig;
const local = localStorage.getItem("config");
if (local !== null) {
    const parsed = JSON.parse(local);
    if (parsed.version === defaultConfig.version) config = parsed;
}

document.addEventListener('keydown', function ({ ctrlKey, shiftKey, code }) {
    if (ctrlKey && shiftKey && code === 'KeyE') {
        if (shown) hideEditor();
        else showEditor();
    }
});

function showEditor() {
    editor.set(config);
    container.classList.remove("hidden");
    shown = true;
}

function hideEditor() {
    config = editor.get(config);
    localStorage.setItem("config", JSON.stringify(config));
    document.getElementById("splash").classList.remove("hidden");
    location.reload();
}

export default config;
