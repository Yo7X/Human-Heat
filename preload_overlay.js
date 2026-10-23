const { contextBridge, ipcRenderer } = require("electron");

document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('sendImagePath').addEventListener('click', () => {
        const path = 'C:/Users/Star/Downloads/' + document.getElementById('imagePath').value
        ipcRenderer.invoke('censorPath', path)
    })
})

contextBridge.exposeInMainWorld("censorAPI", {
    onDetections: (callback) => {
        ipcRenderer.on("detections", (_event, hits) => {
            callback(hits)
        })
    }
})