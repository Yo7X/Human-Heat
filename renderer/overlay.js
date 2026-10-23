window.censorAPI.onDetections((hits) => {
    document.getElementById('imageeee').src = 'C:\\Users\\Star\\AppData\\Roaming\\censor\\screenshot.png'

    document.querySelectorAll('.detection-box').forEach(el => el.remove())

    for (const hit of hits) {
        const aa = document.createElement('div')
        aa.className = 'detection-box'

        aa.style = `
            position: absolute;
            background: rgba(255, 0, 0, 0.5);
            top: ${hit.bounds.y}px;
            left: ${hit.bounds.x}px;
            width: ${hit.bounds.width}px;
            height: ${hit.bounds.height}px;
            pointer-events: none;
        `

        document.body.appendChild(aa)
    }
})