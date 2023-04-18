# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np


class CentroidTracker:
    def __init__(self, maxDisappeared=50, maxDistance=50):
        # khởi tạo ID đối tượng duy nhất tiếp theo cùng với hai thứ tự
        # từ điển được sử dụng để theo dõi ánh xạ một đối tượng nhất định
        # ID đến trọng tâm của nó và số khung hình liên tiếp mà nó có
        # lần lượt được đánh dấu là "biến mất"
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.bbox = OrderedDict()  # CHANGE

        # lưu trữ số lượng khung hình liên tiếp tối đa đã cho
        # đối tượng được phép đánh dấu là "biến mất" cho đến khi chúng tôi
        # cần hủy đăng ký đối tượng khỏi theo dõi
        self.maxDisappeared = maxDisappeared

        # lưu trữ khoảng cách tối đa giữa các trọng tâm để liên kết
        # một đối tượng -- nếu khoảng cách lớn hơn mức tối đa này
        # khoảng cách chúng ta sẽ bắt đầu đánh dấu đối tượng là "biến mất"
        self.maxDistance = maxDistance

    def register(self, centroid, inputRect):
        # khi đăng ký một đối tượng, chúng tôi sử dụng đối tượng có sẵn tiếp theo
        # ID để lưu trữ centroid
        self.objects[self.nextObjectID] = centroid
        self.bbox[self.nextObjectID] = inputRect  # CHANGE
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # để hủy đăng ký ID đối tượng, chúng tôi xóa ID đối tượng khỏi
        # cả hai từ điển tương ứng của chúng tôi
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.bbox[objectID]  # CHANGE

    def update(self, rects):
        # kiểm tra xem danh sách các hình chữ nhật hộp giới hạn đầu vào
        # trống
        if len(rects) == 0:
            # lặp qua bất kỳ đối tượng được theo dõi hiện có nào và đánh dấu chúng
            # như đã biến mất
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                # nếu chúng tôi đã đạt đến số lần liên tiếp tối đa
                # khung nơi một đối tượng nhất định đã được đánh dấu là
                # thiếu, hủy đăng ký
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            # quay lại sớm vì không có centroid hoặc thông tin theo dõi
            # cập nhật
            # trả về self.objects
            return self.bbox

        # khởi tạo một mảng trọng tâm đầu vào cho khung hình hiện tại
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        inputRects = []
        # lặp qua các hình chữ nhật hộp giới hạn
        for (i, (startX, startY, endX, endY, idx, confidence)) in enumerate(rects):
            # sử dụng tọa độ hộp giới hạn để lấy trọng tâm
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)
            inputRects.append(rects[i])  # CHANGE

        # nếu chúng tôi hiện không theo dõi bất kỳ đối tượng nào, hãy nhập dữ liệu vào
        # centroid và đăng ký từng người trong số họ
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i], inputRects[i])  # CHANGE

        # mặt khác, hiện đang theo dõi các đối tượng nên chúng tôi cần
        # cố gắng khớp trọng tâm đầu vào với đối tượng hiện có
        # trọng tâm
        else:
            # lấy bộ ID đối tượng và trọng tâm tương ứng
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # tính khoảng cách giữa mỗi cặp đối tượng
            # trọng tâm và trọng tâm đầu vào tương ứng -- của chúng tôi
            # mục tiêu sẽ là khớp một trọng tâm đầu vào với một trọng tâm hiện có
            # trọng tâm đối tượng
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # để thực hiện so khớp này, chúng ta phải (1) tìm
            # giá trị nhỏ nhất trong mỗi hàng và sau đó (2) sắp xếp hàng
            # chỉ mục dựa trên các giá trị tối thiểu của chúng để hàng
            # với giá trị nhỏ nhất ở *đầu* của chỉ mục
            # danh sách

            rows = D.min(axis=1).argsort()

            # tiếp theo, chúng tôi thực hiện quy trình tương tự trên các cột bằng cách
            # tìm giá trị nhỏ nhất trong mỗi cột và sau đó
            # sắp xếp bằng cách sử dụng danh sách chỉ mục hàng được tính toán trước đó

            cols = D.argmin(axis=1)[rows]

            # để xác định xem chúng tôi có cần cập nhật, đăng ký,
            # hoặc hủy đăng ký một đối tượng mà chúng ta cần theo dõi
            # trong số các chỉ mục hàng và cột mà chúng tôi đã kiểm tra
            usedRows = set()
            usedCols = set()

            # lặp qua sự kết hợp của chỉ mục (hàng, cột)
            # bộ dữ liệu
            for (row, col) in zip(rows, cols):
                # nếu chúng tôi đã kiểm tra hàng hoặc
                # cột giá trị trước, bỏ qua nó
                if row in usedRows or col in usedCols:
                    continue

                # nếu khoảng cách giữa các trọng tâm lớn hơn
                # khoảng cách tối đa, không liên kết hai
                # trọng tâm cho cùng một đối tượng
                if D[row, col] > self.maxDistance:
                    continue

                # mặt khác, lấy ID đối tượng cho hàng hiện tại,
                # đặt trọng tâm mới của nó và đặt lại trọng tâm đã biến mất
                # phản đối
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.bbox[objectID] = inputRects[col]  # CHANGE
                self.disappeared[objectID] = 0

                # chỉ ra rằng chúng tôi đã kiểm tra từng hàng và
                # chỉ mục cột tương ứng
                usedRows.add(row)
                usedCols.add(col)

            # tính cả chỉ số hàng và cột mà chúng ta CHƯA có
            # khám
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # trong trường hợp số trọng tâm của đối tượng là
            # bằng hoặc lớn hơn số trọng tâm đầu vào
            # chúng ta cần kiểm tra xem một số đối tượng này có
            # có khả năng biến mất
            if D.shape[0] >= D.shape[1]:
                # lặp qua các chỉ mục hàng không sử dụng
                for row in unusedRows:
                    # lấy ID đối tượng cho hàng tương ứng
                    # chỉ mục và tăng bộ đếm biến mất
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    # kiểm tra xem số liên tiếp
                    # khung đối tượng đã được đánh dấu "biến mất"
                    # đối với chứng quyền hủy đăng ký đối tượng
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            # mặt khác, nếu số lượng trọng tâm đầu vào lớn hơn
            # hơn số lượng trọng tâm đối tượng hiện có mà chúng ta cần
            # đăng ký mỗi trọng tâm đầu vào mới làm đối tượng có thể theo dõi
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col], inputRects[col])

        # return the set of trackable objects
        # return self.objects
        return self.bbox

