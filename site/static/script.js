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

const loadAge = () => {
    var dob = new Date("06/25/2005")
    var month_diff = Date.now() - dob.getTime()
    var age_dt = new Date(month_diff)
    var year = age_dt.getUTCFullYear()
    var age = Math.abs(year - 1970)
    document.getElementById('age').innerText = age
}

const addField = () => {
    const container = document.getElementById('scores-container')
    const n = container.childElementCount + 1
    container.innerHTML += `<div class="flex flex-row center">
        <p style="font-size: 18px; min-width: 20px;">${n}</p>
        <div class="input-group mb-3">
            <label class="input-group-text" for="inputGroupSelect01">Тип ответа</label>
            <select class="form-select" id="inputGroupSelect01">
                <option value="0" checked>Числовой</option>
                <option value="1">Строка</option>
            </select>
        </div>
        <div class="input-group mb-3">
            <label class="input-group-text" for="inputGroupSelect01">Количество баллов</label>
            <select class="form-select" id="inputGroupSelect01">
                <option value="1" checked>Один</option>
                <option value="2">Два</option>
                <option value="3">Три</option>
            </select>
        </div>
    </div>`
}

const saveExam = async () => {
    const container = document.getElementById('scores-container')
    const exam_id = document.getElementById('exam_id').value
    if (exam_id === '') {
        alert('Укажите ID экзамена')
        return
    }
    const n = container.childElementCount
    let scores = Array(n)
    for (let i = 0; i < n; i += 1) {
        const type = parseInt(container.children[i].children[1].children[1].value)
        const value = parseInt(container.children[i].children[2].children[1].value)
        const pair = [value, type]
        scores[i] = pair
    }
    try {
        const res = await fetch('http://127.0.0.1:5000/add-exam', {
            method: 'POST',
            headers: {
                'accept': 'application/json',
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                'exam_id': exam_id,
                'scores': scores
            })
        });
        const data = await res.json()
        alert(data.message)
    } catch (e) {
        alert('error')
    }
}

const deleteExam = async (examId) => {
    try {
        const res = await fetch(`http://127.0.0.1:5000/delete-exam?exam_id=${parseInt(examId)}`, {
            method: 'POST',
            headers: {
                'accept': 'application/json',
                'content-type': 'application/x-www-form-urlencoded'
            }
        });
        const data = await res.json()
        alert(data.message)
    } catch (e) {
        alert('error')
    }
}