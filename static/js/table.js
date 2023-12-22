let data_table = []
const table_content = document.getElementById('abc')
const fetch_data = async() =>{
    const response = await fetch('/data_table')

    data_table = await response.json();
    console.log(data_table);
    let content = ''
    for(let data of data_table){
        content+=`<tr>
        <td>${data['ten']}</td>
        <td>${data['msv']}</td>
        <td>${data['ngaysinh']}</td>
        <td>${data['diachi']}</td>
        <td>${data['lop']}</td>
    </tr>`
    }
    table_content.innerHTML = content
}

setInterval(fetch_data, 500);