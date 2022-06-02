# Neural_Network_digit_recognition
Mô tả đề tài
Ở đề tài này chúng ta sẽ phải nhận diện được các chữ số từ 0 đến 9, là các chữ số viết tay từ nhiều người khác nhau. Sau đó sắp xếp chúng vào các thư mục được đánh số đúng với số mà ảnh biểu diễn.
Để giải quyết đề bài trên, chúng ta sẽ xây dựng một mạng nơ ron 3 lớp (XX-XX-10)
Vì sao ở lớp đầu vào là XX? Vì bài toán ở đây là một ảnh nên chúng ta chưa biết được lớp đầu vào chính xác của mình là bao nhiêu. (Mở rộng hơn so với yêu cầu đặt ra là 32).
Mô tả dữ liệu từ Kaggle: Kaggle cung cấp cho chúng ta 2 bộ ảnh, một bộ để train và một bộ để test:
	Bộ train gồm 42.000 ảnh gray 28x28px và mỗi pixel sẽ có giá trị từ 0 đến 255 với không là màu đen hoàn toàn và 255 là màu trắng hoàn toàn, và một giá trị đại diện cho con số  mà ảnh đó biểu diễn. Bộ ảnh này được lưu trữ chung trên một file .csv (Comma Separated Values).
	Bộ test gồm 28.000 ảnh gray 28x28px và mỗi pixel sẽ có giá trị từ 0 đến 255 với không là màu đen hoàn toàn và 255 là màu trắng hoàn toàn, và không có giá trị mà chúng ta chỉ sử dụng nó để test (đây là bộ mà chúng ta sẽ thực hiện phân chia). Bộ ảnh này được lưu trữ chung trên một file .csv (Comma Separated Values).
Cơ sở toán học
Trước hết ta xác định rằng đây là một bài toán nhị phân, kết quả đầu ra của bài toán chính là, có hay không bức ảnh đầu vào của chúng ta biểu thị cho một con số cụ thể nào đó, Vì vậy ở lớp đầu ra của chúng ta, node thứ i chính là khả năng ảnh đó có biểu thị cho số i hay không? Con số nầy nằm trong khoảng (0,1) với không là chắc chắn không thể, và 1 là hoàn toàn chắc chắn.
Với bài toán như phân loại nhị phân chúng ta áp dụng Hồi quy logistic (Logistic regression), để ước tính các tham số cho mô hình chúng ta, sao cho kết quả đạt được là tốt nhất.
Bài toán đặt ra
Cho x ϵR^(n_x ), đặt y ̂= P(y=1 ┤|  x)
Chúng ta đặt các tham số sau: w∈R^nx  và b ∈ R
Đầu ra lúc này sẽ là: y ̂= w^T x+b . Tuy nhiên y ̂ là một hàm xác suất nên kết quả của nó phải nằm trong khoảng [0,1] 
Nên đầu ra mong muốn của chúng ta sẽ sử dụng một hàm đặc biệt gọi là sigmoid,
     Ký hiệu là (σ) nên đầu ra lúc này của chúng ta sẽ là
      y ̂=σ( w^T x+b)

 
Hình 3 1 Đồ thị hàm sigmoid
Hàm sigmoid sẽ ép các giá trị nằm trong khoảng (0,1)
Nhưng trong bài toán này chúng ta sẽ không sử dụng hàm sigmoid, thay vào đó chúng ta sẽ sử dụng 2 hàm khác, vấn đề này sẽ được để cập đến ở phần sau.
Hàm chi phí (cost function)
Cho một tập {(x1,y1), (x2,y2) , … , (xm,ym)}. Chúng ta muốn rằng y ̂  ≈ y
Vậy hàm log error function: L(y, y ̂  ) = 1/2 (y ̂-y)^2 hàm này có dạng:
 
Tuy nhiên trong hồi quy logistic hàm  không lồi này, sẽ ảnh hưởng đến thuật toán Gradinet Descent (là thuật toán mà chúng ta dùng để tối ưu tham số, sẽ được đề cập đến sau). Vậy nên chúng ta xây dựng một hàm log errors mới có  dạng như sau
L(y, y ̂  ) = -(y log⁡y ̂ +(1-y)  log⁡(1-y ̂ ) )
	Nếu y = 1,  L(y, y ̂  ) = - y log⁡y ̂  muốn logy ̂ lớn => y ̂  lớn
	Nếu y = 0,  L(y, y ̂  ) = (1- y)  〖log⁡(1-〗⁡〖y ̂)〗 muốn logy ̂ lớn => y ̂  nhở
Hàm chi phí : J = 1/m ∑_(i=1)^m▒L(y ̂_i,y_i ) 
Đồ thị: nếu y = 1
 
Đồ thị nếu y = 0:
 
Gradien Descent
Khi đã xác định được hàm chi phí J, điều chúng ta cần tìm chính là tìm các tham số đầu vào w và b sao cho J(w,b) là bé nhất, đẫn đến một bài toán tìm cực tiểu, và Gradient Descent chính là thuật toán làm điều đó.
 
Và chúng ta muốn hàm điểm của chúng ta tìm được chính là hàm lồi có dạng như trên, đó là lý do vì sao hàm log error của chúng ta, không thể làm hàm bình phương vì nó sẽ có dạng tương tự thế này.
 
Các điểm được chỉ vào là các điểm trũng gọi là các điểm tối ưu cục bộ, cái chúng ta đang tìm là điểm tối ưu toàn cục nên việc có một hàm log errors mới là cần thiết.
Thuật toán Gradient Descent được trình bày như sau:
 
Bước 1: Từ một điểm w khởi tạo ban đầu:
Bước 2: w = w – α ⅆJ(w)/ⅆw
Bước 3: Lặp lại quá trình trên đến khi tìm được w tối ưu 
Trong đó  α được gọi là learning rate, là tham số mà chúng ta đặt ra sao cho bài toán đạt được là tối ưu nhất.
ⅆJ(w)/ⅆw   đại diện cho đạo hàm của J(w) tại điểm w biểu thị cho độ dốc tại điểm đó. Độ dốc càng lớn, tốc độ trở về điểm tối ưu càng nhanh. Quay trở lại hàm signoid.
 
Như đã đề cập ở trên, hàm signoid không tối ưu cho thuật toán của cúng ta. Nhìn vào đồ thị trên ta có thể thấy khi x, càng lớn, f’(x) sẽ càng tiến về không, nghĩa là tốc độ của thuật toán sẽ giảm. Vậy nên, để khắc phục nhược điểm trên một hàm mới được sinh ra đó chính là hàm ReLU
 
Trong bài này, chúng ta cũng sẽ sử dụng hàm ReLU.
Đồ thị tính toán (Computation Graph)
Ví dụ về một đồ thị tính toán:
 
Với đồ thị tính toán trên  ta có thể lấy được các đạo hạm bất kỳ ví dụ
ⅆ_J/ⅆv=3,  ⅆ_J/ⅆu=ⅆ_J/ⅆv⋅ⅆv/ⅆa  Tương tự, ta có thể tính các đạo hàm trước, dựa vào cá đạo hàm sau đã được tính rồi.
Dựa vào mô hình trên ta có sơ đồ sau
 
Theo cách truyền ngược  như trên ta có:
"ⅆa"=dL(a,y)/da= -y/a+(1-y)/(1-a)
"ⅆ" z=ⅆ_L/ⅆa⋅ⅆa/ⅆz  = a- y (với ⅆa/ⅆz= a(1-a))
Việc triển khai Gradient Descent sẽ trên m mẫu thử có thể thực hiện bằng cách khởi tạo các biến J, dw, db, sau đó lần lượt tính toán trên từng giá trị, x ∈  R^n chúng ta buộc  phải thực hiện thêm một vòng lặp để có được giá trị đó. Độ phức tạp lúc này đã là O(m*n) , điều này là không tốt chút nào với  deep learning nơi mà lượng dữ liệu đầu vào cực lơn.
Câu hỏi đặt ra là liệu có cách nào để chúng ta có thể tính toán các giá trị đó một lượt hay không? 
Vector hóa(Vectorization)
Vector hóa là kỹ thuật quan trọng trong Deep learning, và để thực hiện được điều này, chúng ta sẽ cần đến một môn học khác ngoài giải tích, chính là đại số tuyến tính. 
Với x ϵR^n chúng ta sẽ biểu diễn x dưới dạng một vector n chiều, lúc này chúng ta đã loại bỏ được một vòng lặp phía trong, tiếp đến, chúng ta sẽ sếp m vector x vào các cộ, có tổng cộng là m vector x, nên ta sẽ được ma trận là nxm
Ma trận có dạng như sau
|	|	|	|
X1	X2	…	Xm
|	|	|	|

 Câu hỏi đặt ra là, liệu có cách nào để tính toán ma trận trên nhanh hơn cách thông thường không?
Rất may là thư viện numpy của python đã có các hàm dựng sẵn (built-in function) dùng để tính toán các ma trận cực kỳ nhanh chóng. Hãy làm một phép đo đơn giản.
 
Với thư viện numpy tính toán trên vector, thời gian thực hiện phép nhân là 2.5ms
Với cách tính thông thường, thời gian thực hiện là  485.7ms 
Cách tính theo vector cho hiệu suất tốt hơn đến ≈19328%
Như vậy chúng ta đang đi đúng hướng.
Mạng nơ ron (Neural network)
Ý tưởng
Từ mô hình toán học được đề cập ở trên
 
Chúng ta mô hình hóa nó như sau:
 
Để tính toán trên mạng này chúng ta cần mở rộng phần lớp thứ 2 gọi là lớp ẩn, Ta có mạng nơ ron đơn giản sau.
 
Tính toán theo chiều xuôi, chúng ta sẽ được như sau
x	Layer[1]	Layer[2]	Log
W	z[1] =W[1]*x+b	a[1] = g(z[1])	z[2]=W[2]*a[1]+b	a[2] = g(z[2])	L(a[2],y)
b	Chiều xuôi 

Trình bày vè mạng nơ ron

 
Mạng  nơ ron trong bài toán nhận diện số
Trong bài toán này, chúng ta sẽ xây dựng mạng nơ ron để hiện thực bài toán nhận diện số. Ảnh đầu vào của chúng ta là ảnh 28x28 pixel mỗi pixel có giá trị từ 0-255. 
Nên chúng ta sẽ xếp các pixel thành một cột duy nhất có 28x28 = 784
	Vậy Layer 0  của chúng ta sẽ có 784 nơ ron
	Để xử lý bài toán trên ở Layer 1 chúng ta cũng có 10 nơ ron
	Tất nhiên  Layer 2 (output) sẽ có 10 nơ ron đại diện cho kết quả ta vừa tính được.
Mạng nơ ron của chúng ta sẽ có dạng như sau:
Layer 0 (input)	Layer 1 (hidden)	Layer 3 (output)	Result
0
0
0

0
0
0
0
..
..
..	
0	0
0	
784	10	10	1

Xếp các lớp này theo hàng dọc, ta được ma trận đầu vào (mxn) như sau:
X11	X21	…	Xm1
|	|	|	|
|	|		|
|	|	|	|
X1n	X2n	…	Xmn

A = X (nxm)
Từ các công thức  toán học ở trên  ta rút ra được bộ công thức sau:
Công thức truyền xuôi:
 
hàm hiệu chỉnh lúc này sẽ được thay đổi vì kết quả đầu ra mong muốn của chúng ta nằm trong khoảng 0 dến 1,  tuy  nhiên hàm signoid không phải là một sự lựa chọn tốt vậy nên chúng ta sẽ sử dụng một hàm mới gọi là hàm softmax có công thức như sau: 
 
Công thức truyền ngược:









