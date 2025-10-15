#1. Tạo một ma trận 3x3 có tên A với các phần tử

#2. Chuyển đổi ma trận A thành kiểu số nguyên và lưu vào ma trận B. In ra ma #trận B và kiểu dữ liệu của ma trận B.

#3. Tạo một vector u với giá trị $[0.5, 1.5, 2.5]$ và kiểu dữ liệu là #`float64`. Cộng ma trận B với vector u và lưu kết quả vào ma trận C. In ra ma #trận C và kiểu dữ liệu của ma trận C.

#4. Tạo một vector v với giá trị $[10, 20, 30]$ và kiểu dữ liệu là `int32`. #Cộng ma trận B với vector v và lưu kết quả vào ma trận D. In ra ma trận D và #kiểu dữ liệu của ma trận D.

# Nhập thư viện numpy với bí danh np
import numpy as np
# 1. Tạo một mảng numpy có tên A
A = np.array([
    [5.1, 8.7, 3.3],
    [4.4, 0.9, 9.5],
    [6.2, 7.5, 1.8]
])

# 2. Ma trận B
B = A.astype(int)
print(B, B.dtype)

# 3. Ma trận C
u = np.array([0.5, 1.5, 2.5],dtype=np.float64)
C = B + u
print(C, C.dtype)
# 4. Ma trận D
v = np.array([10, 20, 30],dtype=np.int32)
D = B + v
print(D, D.dtype)

#1. Tạo một ma trận 6x6 có tên M, chứa các số từ 0 đến 35.
#2. Trích xuất vùng trung tâm 4x4 của ma trận M. Vùng này bao gồm các hàng từ #1 đến 4 và cột từ 1 đến 4 (Ma trận P).
#3. Chọn các hàng 0 và 3 của ma trận con P.
#4. Gán giá trị -99 cho các cột 1 và 2 của ma trận M.
#5. Trích xuất các giá trị trong các cột lẻ (1, 3, 5) của ma trận M đã bị thay đổi, và tìm các giá trị lớn hơn 20 trong các cột này.

import numpy as np
# 1. Tạo ma trận M
M = np.arange(0, 36).reshape(6, 6)
# 2. Trích xuất vùng trung tâm 4x4
P = M[1:5, 1:5]
# Hàng từ 1 đến 4, cột từ 1 đến 4

# 3. Chọn các hàng 0 và 3 của P
R = P[[0, 3], :]
# 4. Gán giá trị cho các cột 1 và 2 của M
N = M.copy()  
N[:, [1, 2]] = -99
# 5. Trích xuất giá trị theo điều kiện kết hợp
# Lấy ra các cột lẻ từ ma trận đã bị thay đổi
S = M[:, 1::2]
Value = S[S > 20]
print("Ma trận M:\n", M)
print("Vùng trung tâm P:\n", P)
print("Ma trận M sau khi thay đổi:\n", N)
print("Hàng 0 và 3 của P:\n", R)
print("Giá trị lớn hơn 20 trong các cột lẻ của M:\n", Value)

#1. Tạo một ma trận `data` kích thước 4x5 với các giá trị ngẫu nhiên trong #khoảng từ 10 đến 100. In ra ma trận `data`.
#2. Tính ma trận chuyển vị của `data` và in ra kết quả.
#3. Tính giá trị lớn nhất và nhỏ nhất cho từng hàng trong ma trận `data`. In #ra kết quả.
#4. Tìm chỉ số của cột chứa giá trị lớn nhất trong mỗi hàng của ma trận #`data`. In ra kết quả.

import numpy as np
# 1. Tạo ma trận data
data = np.random.randint(10, 101, size=(4, 5))
print("Ma trận data:\n", data)
# 2. Chuyển vị
data_T = data.T
print("Ma trận data_T:\n", data_T)
# 3. Tính max và min cho từng hàng
max_row = np.max(data, axis=1)
min_row = np.min(data, axis=1)
print("Giá trị lớn nhất trong mỗi hàng:\n", max_row)
print("Giá trị nhỏ nhất trong mỗi hàng:\n", min_row)
# 4. Tìm chỉ số của giá trị lớn nhất trong mỗi hàng
max_indices = np.argmax(data, axis=1)
print("Chỉ số của giá trị lớn nhất trong mỗi hàng:\n", max_indices)

#1. Tạo một ma trận chi phí có tên `costs` với các phần tử như sau:
#   * Dòng 1: Chi phí nguyên vật liệu, nhân công, vận hành cho sản phẩm A.
#   * Dòng 2: Chi phí nguyên vật liệu, nhân công, vận hành cho sản phẩm B.
#  * Dòng 3: Chi phí nguyên vật liệu, nhân công, vận hành cho sản phẩm C.
#2. Tạo một vector `production_volume` chứa số lượng sản phẩm A, B, C được sản #xuất lần lượt là 1000, 500, 1500.
#3. Tính tổng chi phí cho từng loại chi phí (Nguyên vật liệu, Nhân công, Vận #hành).
#4. Tính tổng chi phí cho từng sản phẩm (trên một đơn vị sản phẩm).

import numpy as np
# 1. Tạo ma trận chi phí
costs = np.array([
    [10, 5, 2],    # Sản phẩm A
    [12, 7, 3],    # Sản phẩm B
    [9, 4, 2.5]    # Sản phẩm C
])
print("Ma trận chi phí (Sản phẩm x Loại chi phí)")
print(costs)

# 2. Tạo vector số lượng
product_volume = np.array([1000, 500, 1500]) 
# 3. Tính tổng chi phí cho từng loại
total_cost_by_type = np.dot(product_volume, costs)
print("Tổng chi phí cho từng loại (Nguyên liệu, Nhân công, Vận chuyển):")
print(total_cost_by_type)
# 4. Tính tổng chi phí cho từng sản phẩm (trên 1 đơn vị)
total_cost_by_product = np.sum(costs*product_volume, axis=1)
print("Tổng chi phí cho từng sản phẩm (trên 1 đơn vị):")
print(total_cost_by_product)

#1. Tạo một mảng `x` từ -3 đến 3 với bước nhảy là 0.5. In ra mảng `x` sau khi #làm tròn đến 2 chữ số thập phân.
#2. Tính giá trị hàm Sigmoid cho từng phần tử trong mảng `x` và in kết quả ra #mảng mới. In ra mảng kết quả sau khi làm tròn đến 4 chữ số thập phân.


# 1. Tạo mảng x
x = np.arange(-3, 3.5, 0.5)
print("Mảng x:", x)
x_rounded = np.round(x, 2)
print("Làm tròn 2 chữ số:", x_rounded)
# 2. Tính hàm Sigmoid
sigmoid = 1 / (1 + np.exp(-x))
print("Sigmoid:", sigmoid)
sigmoid_rounded = np.round(sigmoid, 4)
print("Làm tròn 4 chữ số:", sigmoid_rounded)


#1. Tạo một ma trận 3x3 có tên A và một vector b

#2. Tính định thức của ma trận A và kiểm tra xem ma trận A có khả nghịch hay #không.

#3. Tính ma trận nghịch đảo của A (A^{-1}).

#4. Tìm nghiệm của hệ phương trình $A \cdot x = b$ bằng công thức $x = A^{-1} #\cdot b$.

#5. Kiểm tra lại nghiệm x bằng cách nhân lại với ma trận A và so sánh kết quả #với vector b ban đầu.


import numpy as np

# 1. Tạo ma trận 3x3 có tên A
A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 19]
])
print(f"Ma trận A: \n{A}")

b = np.array([10, 20, 30])
print(f"Vector b: {b}")

# 2. Tính định thức của A
det_A = np.linalg.det(A)
print(f"Định thức của A: {det_A}")
# 3. Tính ma trận nghịch đảo A^{-1}
if det_A != 0:
    inv_A = np.linalg.inv(A)
else:
    inv_A = None
print(f"Ma trận nghịch đảo A^(-1): \n{inv_A}")
# 4. Tìm nghiệm x bằng công thức x = A^{-1}b
# Sử dụng toán tử @ để nhân ma trận
if inv_A is not None:
    x = inv_A @ b
else:
    x = None
print(f"Nghiệm x: {x}")
# 5. Kiểm tra lại kết quả
if x is not None:
    check = A @ x
else:
    check = None
print("A @ x = ", check)
print("Vector b = ", b)
print("sai so: ", np.abs(check - b))


#1. Tạo một Series từ Dictionary với các thành phố và dân số của chúng như #sau:
#   * "Hanoi": 8.5 triệu
#   * "HCMC": 9.3 triệu
#  * "Danang": 1.2 triệu
#  * "Haiphong": 2.1 triệu

#2. Trích xuất dân số của Hà Nội và Đà Nẵng từ Series và in ra.
#3. Thực hiện phép toán vector hóa để chuyển đổi dân số các thành phố từ triệu #người sang người (nhân với 1 triệu), sau đó in kết quả ra dưới dạng số #nguyên.

import pandas as pd
# 1. Tạo Series từ Dictionary
population_dict = {"Hanoi": 8.5, "HCMC": 9.3, "Danang": 1.2, "Haiphong": 2.1}
city_pop = pd.Series(population_dict)
print("Dân số các thành phố (triệu người)")
print(city_pop)

# 2. Trích xuất dữ liệu
hanoi_pop = city_pop["Hanoi"]
print(f"Dân số Hà Nội: {hanoi_pop} triệu người")
danang_pop = city_pop["Danang"]
print(f"Dân số Đà Nẵng: {danang_pop} triệu người")
# 3. Thực hiện phép toán vector hóa
city_people = city_pop * 1000000
print(city_people.astype(int))

#1. Tạo một DataFrame có tên `sales_df` với các cột sau:

#   * `nhan_vien`: Danh sách tên nhân viên ("An", "Binh", "Chi", "Dung").
#   * `khu_vuc`: Danh sách khu vực làm việc của các nhân viên ("Bắc", "Nam", #"Trung", "Bắc").
#   * `doanh_so`: Danh sách doanh số bán hàng của các nhân viên (120, 250, #180, 90).
#2. Thêm cột "thuong" vào DataFrame. Cột "thuong" có giá trị True nếu doanh số #lớn hơn 100, ngược lại là False.
#3. Thêm cột "tien\_thuong" vào DataFrame. Cột "tien\_thuong" tính toán tiền #thưởng cho nhân viên. Nếu cột "thuong" là True, tiền thưởng sẽ bằng 10% doanh #số bán hàng, nếu không sẽ là 0.
#4. Xóa cột "khu\_vuc" khỏi DataFrame.

import pandas as pd

# 1. Tạo DataFrame
sales_data = {
    "nhan_vien": ["An", "Binh", "Chi", "Dung"],
    "khu_vuc": ["Bắc", "Nam", "Trung", "Bắc"],
    "doanh_so": [120, 250, 180, 90]
}
sales_df = pd.DataFrame(sales_data)
print("DataFrame Bán hàng ban đầu")
print(sales_df)

# 2. Thêm cột "thuong" dựa trên điều kiện
sales_df["thuong"] = sales_df["doanh_so"] >100
# 3. Thêm cột "tien_thuong"
# Sử dụng np.where(điều_kiện, giá_trị_nếu_đúng, giá_trị_nếu_sai)
sales_df["tien_thuong"] = np.where(sales_df["thuong"], sales_df["doanh_so"] * 0.1, 0)
print(sales_df)
# 4. Xóa cột "khu_vuc"
sales_df = sales_df.drop(columns=["khu_vuc"])
print(sales_df)

#1. Tạo một DataFrame với dữ liệu từ 0 đến 15, có kích thước 4x4. Đặt tên cho #các chỉ mục hàng là "A", "B", "C", "D" và tên các cột là "W", "X", "Y", "Z".
#2. Lựa chọn dữ liệu từ hàng "A" và "C", cột "W" và "Y" bằng cách sử dụng #`.loc` (theo nhãn).
#3. Lựa chọn dữ liệu từ 2 hàng cuối (vị trí 2 và 3) và 2 cột đầu (vị trí 0 và #1) bằng cách sử dụng `.iloc` (theo vị trí số nguyên).
#4. Lấy giá trị tại hàng "B", cột "Z".

import pandas as pd
import numpy as np

# 1. Tạo DataFrame
df = pd.DataFrame(
    np.arange(16).reshape(4, 4),
    index=["A", "B", "C", "D"],
    columns=["W", "X", "Y", "Z"]
)
print("DataFrame gốc")
print(df)

# 2. Lựa chọn bằng .loc (theo nhãn)
selected_loc = df.loc[["A", "C"], ["W", "Y"]]
print("Lựa chọn bằng .loc (hàng A, C và cột W, Y):")
print(selected_loc)
# 3. Lựa chọn bằng .iloc (theo vị trí số nguyên)
selected_iloc = df.iloc[[2, 4], [0, 2]]
print(selected_iloc)
# 4. Lấy một giá trị đơn lẻ
value = df.loc["B", "X"]
print(f"Giá trị tại hàng B và cột X: {value}")

#1. Tạo một Series có tên `product_stock` với các giá trị `[10, 50, 30]` và #chỉ số `[0, 3, 5]`.
#2. Sử dụng phương thức `reindex` để tạo ra các chỉ số từ 0 đến 5. In ra #Series sau khi reindex.
#3. Sử dụng phương thức `reindex` và điền giá trị bị thiếu bằng phương thức #`ffill` (forward-fill). In ra Series sau khi reindex và điền giá trị.

import pandas as pd
import numpy as np

product_stock = pd.Series([10, 50, 30], index=[0, 3, 5])
print("Series ban đầu")
print(product_stock)

# 1. Reindex để tạo ra các chỉ số bị thiếu
reindexed = product_stock.reindex(range(6))
print("Sau khi reindex:")
print(reindexed)
# 2. Reindex và điền giá trị bằng 'ffill' (forward-fill)
filled = product_stock.reindex(range(6), method='ffill')
print("Sau khi reindex với phương pháp 'ffill':")
print(filled)

#1. Tạo một DataFrame với thông tin điểm số của các sinh viên (môn Toán, Lý, #Hóa) với các sinh viên là Hùng, Lan, Minh, An. In ra bảng điểm ban đầu.
#2. Sắp xếp bảng điểm theo điểm môn Toán giảm dần. In ra bảng điểm sau khi sắp #xếp.
#3. Tính tổng điểm của mỗi sinh viên và thêm cột tổng điểm vào bảng điểm. In #ra bảng điểm với cột tổng điểm.
#4. Thực hiện thống kê mô tả cho các môn Toán, Lý và Hóa. In ra kết quả thống #kê mô tả.

import pandas as pd
import pandas as pd

# 1. Tạo DataFrame
scores_data = {
    "Toan": [7, 5, 9, 8],
    "Ly": [8, 9, 6, 7],
    "Hoa": [6, 8, 9, 5]
}
student_scores = pd.DataFrame(scores_data, index=["Hùng", "Lan", "Minh", "An"])
print("Bảng điểm ban đầu")
print(student_scores)

# 2. Sắp xếp theo điểm Toán giảm dần
sorted_by_math = student_scores.sort_values(by="Toan", ascending=False)
print(sorted_by_math)
# 3. Tính tổng điểm mỗi sinh viên
student_scores["Tong"] = student_scores.sum(axis=1)
print(student_scores)
# 4. Thống kê mô tả
print(student_scores.describe())

#1. Đọc file `transactions.csv` vào DataFrame
#2. Hiển thị thông tin cơ bản về DataFrame
#   * In ra 5 dòng đầu tiên của DataFrame.
#   * Hiển thị thông tin và kiểu dữ liệu của DataFrame.
#3. Điền giá trị thiếu trong cột "SoLuong"
#   * Điền giá trị thiếu bằng 1 cho cột "SoLuong".
#   * Sử dụng `inplace=True` để thay đổi trực tiếp trên DataFrame gốc.
#4. Tạo cột mới "ThanhTien"
#   * Tính giá trị của cột "ThanhTien" bằng cách nhân cột "DonGia" với cột #"SoLuong".
#5. Lọc các giao dịch thuộc nhóm hàng "Điện tử"
#   * Lọc và in ra các giao dịch có nhóm hàng là "Điện tử".


import pandas as pd
import numpy as np

# 1. Đọc file CSV vào DataFrame
# Giả sử file transactions.csv nằm cùng thư mục với code
df = pd.read_csv("transactions.csv")

print("DataFrame sau khi đọc từ file CSV")
print(df)

# 2. Hiển thị thông tin cơ bản
print(df.head())
print(df.info())
# 3. Điền giá trị thiếu trong cột "SoLuong"
df[df["SoLuong"].fillna(1,inplace=false)]
# 4. Tạo cột mới "ThanhTien"
df["ThanhTien"] = df["SoLuong"] * df["DonGia"]
# 5. Lọc các giao dịch thuộc nhóm hàng "Điện tử"
df_electronics = df[df["NhomHang"] == "Điện tử"]
print(df_electronics)

#matplotlib
Import matplotlib.pyplot as plt
plt.plot(x,y,...)
#bar
plt.bar(x.y)
plt.title(‘ten’)
plt.show()
#vd
D={}
plt.bar(range(len(D)),D.value(),align=’center)
plt.xticks(range(len(D)),D.key())
plt.title()
plt.show()
#chồng cột bottom=
#chia cột x+-width/2
#pie
#plt.pie(x,labels,autopct=’%1.1f%%’,explode,startangle,shadow,radius)
#plt.pie(d.values(),labels=d.keys())
#plt.pie(df[],labels=df[])
#scatter phân tán
#stackplot ngăn xếp
#grid lưới 
#subplot(3,1,1)
#imread (image)
#savefig (lưu ảnh ra file)

