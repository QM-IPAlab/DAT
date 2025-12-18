// 页面加载完成后执行
document.addEventListener('DOMContentLoaded', function() {
    // 可以添加图片轮播等功能
    console.log('DAT Project Website Loaded');
    
    // 示例：为所有表格添加斑马纹
    const tables = document.querySelectorAll('table');
    tables.forEach((table, index) => {
        if (index % 2 === 0) {
            table.classList.add('is-striped');
        }
    });
});
