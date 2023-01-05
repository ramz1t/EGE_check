const fileTypes = [
    "image/apng",
    "image/bmp",
    "image/gif",
    "image/jpeg",
    "image/pjpeg",
    "image/png",
    "image/svg+xml",
    "image/tiff",
    "image/webp",
    "image/x-icon"
]

function validFileType(file) {
    return fileTypes.includes(file.type)
}

function updateImageDisplay() {
    const input = document.getElementById('files')
    const preview = document.querySelector('.preview')
    while (preview.firstChild) {
        preview.removeChild(preview.firstChild)
    }
    const curFiles = input.files
    if (curFiles.length === 0) {
        const para = document.createElement('p')
        para.textContent = 'No files currently selected for upload'
        preview.appendChild(para)
    } else {
        const list = document.createElement('div');
        list.classList = 'form_list'
        preview.appendChild(list)

        for (const file of curFiles) {
            const listItem = document.createElement('div')
            const para = document.createElement('div');
            if (validFileType(file)) {
                para.classList = "alert alert-secondary"
                para.role = 'alert'
                let filename = file.name
                if (filename.length > 22) {
                    filename = '...' + filename.slice(-19)
                }
                para.textContent = filename
                const image = document.createElement('img')
                image.src = URL.createObjectURL(file)
                image.classList = 'form_image'
                listItem.appendChild(image)
                listItem.appendChild(para)
            } else {
                para.classList = 'alert alert-warning'
                para.role = 'alert'
                para.textContent = `${file.name}: Файл не будет проверен, загрузите изображение`
                listItem.appendChild(para)
            }

            list.appendChild(listItem)
        }
    }
}
