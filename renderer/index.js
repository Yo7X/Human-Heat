let imageRender = null

document.addEventListener('DOMContentLoaded', () => {
    imageRender = document.getElementById('imgDisplay')
})

window.censorAPI.onDetections((hits) => {
    document.querySelectorAll('img').forEach((imgz) => {
        imgz.remove()
    })

    const newImg = document.createElement('img')
    newImg.src = "C:/users/Star/AppData/Roaming/censor/detections.png?t=" + Date.now()
    newImg.classList.add('imageWithBounds')
    document.getElementById('bodyBoundingBoxes').appendChild(newImg)
    newImg.onload = () => {
        console.log(newImg.getBoundingClientRect())
        console.log(newImg.naturalWidth, newImg.naturalHeight)
        document.getElementById('imageInfo').innerHTML = `
        Original image bounds: ${newImg.naturalWidth} x ${newImg.naturalHeight} <br>
        Number of detections: ${hits} <br>
        Required confidence: 0.5 > <br>
        `
    }

    for(let i = 0; i < hits; i++) {
        const newFlexItem = document.createElement('div')
        newFlexItem.classList.add('containerItem')

        const newImg = document.createElement('img')
        newImg.src = `C:/users/Star/AppData/Roaming/censor/body${i}.png?t=${Date.now()}`
        newImg.classList.add('bodyImage')
        newFlexItem.appendChild(newImg)

        const newImg2 = document.createElement('img')
        newImg2.src = `C:/users/Star/AppData/Roaming/censor/bodyHeatMap${i}.png?t=${Date.now()}`
        newImg2.classList.add('bodyImage')
        newFlexItem.appendChild(newImg2)

        document.getElementById('bodyImageContainer').appendChild(newFlexItem)
    }

    
})

