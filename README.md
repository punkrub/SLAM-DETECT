# SLAM และตรวจจับป้าย RoboMaster

## เตรียมโปรเจกต์

Clone พร้อม SLAM และ RoboMaster SDK:

```powershell
git clone --recurse-submodules https://github.com/punkrub/SLAM-DETECT.git
cd SLAM-DETECT
py -3.8 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Windows ต้องมี CMake และ Visual Studio C++ Build Tools เพื่อ build codec สำหรับกล้อง:

```powershell
Push-Location .\RoboMaster-SDK
python setup_with_lib.py install
Pop-Location
```

## รัน

เชื่อมคอมพิวเตอร์กับ Wi-Fi ของหุ่น แล้วสั่ง:

```powershell
python slam_detect_camera.py --conn-type ap
```

โปรแกรมสำรวจ SLAM และตรวจจับ/mark ป้ายในรอบเดียว กด `q` ที่หน้าต่างกล้องเพื่อหยุด

## ไฟล์ผลลัพธ์

แต่ละรอบสร้างโฟลเดอร์ใหม่ที่ `mission_maps/run_<เวลา>/` ภายในมี `explored_map.json` และ `map.png` ซึ่งแสดงแผนที่กับตำแหน่งป้ายที่ mark

`detect.py` ไม่จำเป็น; ตัวตรวจจับอยู่ใน `detect_camera.py` ส่วน SLAM และ SDK ใช้ submodule เดิม