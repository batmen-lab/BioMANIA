const { app, BrowserWindow } = require('electron');
const { exec } = require('child_process');
const path = require('path');

function createWindow() {
  const win = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      nodeIntegration: true
    }
  });

  // 加载前端应用，假设它在 localhost:3000 上运行
  win.loadURL('http://localhost:3000'); 

  // 启动前端服务
  exec('npm run dev', { cwd: path.join(__dirname, 'chatbot_ui_biomania') }, (err, stdout, stderr) => {
    if (err) {
      console.error(`Error starting frontend: ${err}`);
      return;
    }
    console.log(`Frontend started: ${stdout}`);
  });

  // 启动后端服务
  exec('python -m src.deploy.inference_dialog_server', (err, stdout, stderr) => {
    if (err) {
      console.error(`Error starting backend: ${err}`);
      return;
    }
    console.log(`Backend started: ${stdout}`);
  });
}

// 确保在 app 准备好时才创建窗口
app.whenReady().then(() => {
  createWindow();

  app.on('activate', () => {
    // 在 macOS 上，当 dock 图标被点击时，如果没有其他窗口打开，通常会重新创建一个窗口
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  // 在 macOS 上，用户通常会在他们明确地关闭所有窗口时退出应用
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
