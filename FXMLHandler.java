import javafx.fxml.FXML;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import javafx.event.ActionEvent;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.HBox;
import javafx.stage.FileChooser;
import javafx.stage.Stage;

public class FXMLHandler {
	@FXML
	private Button btnTest;
	@FXML
	private Label label;
	@FXML
	private void btnTestMouseClickHandler(ActionEvent event) throws IOException {
		btnTest.setText("TEST");
		label.setText("버튼을 눌렀습니다.");
		FileChooser fileChooser = new FileChooser();
		fileChooser.setTitle("Open Resource File");
		File test = null;
		try {
			test = fileChooser.showOpenDialog(null);
			label.setText(test.getPath());
		} catch (Exception e) {
			System.out.println("TEST");
		}

		houghTransForm(test.getPath(), test.getName());
		Image image = new Image("file:" + test.getName()); // 주의! file:을 붙여야 로컬로 인식한다.
		ImageView imageView = new ImageView(image);
		Stage stage = new Stage();
		Group group = new Group();
		HBox box = new HBox();
		Scene scene = new Scene(group);
		box.getChildren().add(imageView);
		group.getChildren().add(box);
		stage.setScene(scene);
		stage.show();
		stage.sizeToScene();
	}

	private void cannyEdgeDetect(String path, String name) { // 케니 엣지 추출
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		Mat source = Imgcodecs.imread(path, Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
		Imgproc.Canny(source, source, 50, 100);
		Imgcodecs.imwrite(name, source);
	}

	private void houghTransForm(String path, String name) { // 허프 라인 추출
		// 추후 코드 모듈화 예정
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		Mat source = Imgcodecs.imread(path, Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
		Mat dstImage = new Mat();
		Imgproc.cvtColor(source, dstImage, Imgproc.COLOR_GRAY2BGR);
		Imgproc.blur(dstImage, dstImage, new Size(3, 3));

		Mat edges = new Mat();
		Imgproc.Canny(dstImage, edges, 80, 100); // 케니 완료

		Mat lines = new Mat();
		Imgproc.HoughLines(edges, lines, 1, Math.PI / 180.0, 100); // 허프 완료

		double rho, theta;
		double c, s;
		double x0, y0;
		double data[];
		class HoughClass {
			private double rho;
			private double theta;

			public HoughClass(double rho, double theta) {
				this.rho = rho;
				this.theta = theta;
			}
		}
		ArrayList<HoughClass> list = new ArrayList<>();
		System.out.println(lines.rows());

		// 허프 라인 따는 부분
		label_1: for (int i = 0; i < lines.rows(); i++) { // 자바에서는 Rows와 Cols가 다르다.
			data = lines.get(i, 0);
			rho = data[0];
			theta = data[1];
			for (int k = 0; k < list.size(); k++) {
				if (Math.abs(Math.abs(list.get(k).rho) - Math.abs(rho)) < source.width() * 0.1
						&& Math.abs(list.get(k).theta - theta) < Math.PI * 2 * 0.1) {
					continue label_1;
				}
			}
			if (theta == 0)
				theta = 0.00001;
			list.add(new HoughClass(rho, theta));
//			c = Math.cos(theta);
//			s = Math.sin(theta);
//			x0 = c * rho;
//			y0 = s * rho;
//			Point pt1 = new Point(x0 + 1000 * (-s), y0 + 1000 * c);
//			Point pt2 = new Point(x0 - 1000 * (-s), y0 - 1000 * c);
//			Imgproc.line(dstImage, pt1, pt2, new Scalar(0, 0, 255), 1);
//			System.out.println("rho: " + rho + " // theta: " + theta);
		}
		// rho를 기준으로 오름차순 정렬
		java.util.Collections.sort(list, new Comparator<HoughClass>() {
			@Override
			public int compare(HoughClass obj1, HoughClass obj2) {
				return (Math.abs(obj1.rho) < Math.abs(obj2.rho)) ? -1
						: (Math.abs(obj1.rho) > Math.abs(obj2.rho)) ? 1 : 0;
			}
		});
		for (int i = 0; i < list.size(); i++)
			System.out.print("rho: " + list.get(i).rho + " ");
		System.out.println("");
		// 라인 순서 변경
		double r1, r2;
		double theta1, theta2;
		for (int i = 0; i < 4; i = i + 2) {
			r1 = list.get(i).rho;
			r2 = list.get((i + 1) % 4).rho;
			theta1 = list.get(i).theta;
			theta2 = list.get((i + 1) % 4).theta;
			if (r1 < 0 || r2 < 0) {
				if (theta1 > theta2 && theta2 > Math.PI - theta1) {
					list.get(i).rho = r2;
					list.get((i + 1) % 4).rho = r1;
					list.get(i).theta = theta2;
					list.get((i + 1) % 4).theta = theta1;
				}
			} else if (r1 >= 0 && r2 >= 0) {
				if (theta1 < theta2 && theta2 < Math.PI / 2) {
					list.get(i).rho = r2;
					list.get((i + 1) % 4).rho = r1;
					list.get(i).theta = theta2;
					list.get((i + 1) % 4).theta = theta1;
				}
			}
		}
		for (int i = 0; i < list.size(); i++)
			System.out.print("rho: " + list.get(i).rho + " ");
		Point[] pt = new Point[4];
		// x, y 좌표 획득 과정
		for (int i = 0; i < pt.length; i++) {
			pt[i] = new Point();
			r1 = list.get(i).rho;
			r2 = list.get((i + 1) % 4).rho;
			theta1 = list.get(i).theta;
			theta2 = list.get((i + 1) % 4).theta;
			pt[i].x = r2 / Math.sin(theta2) - r1 / Math.sin(theta1);
			pt[i].x /= Math.cos(theta2) / Math.sin(theta2) - Math.cos(theta1) / Math.sin(theta1);
			pt[i].y = (r1 - pt[i].x * Math.cos(theta1)) / Math.sin(theta1);
		}
		System.out.println("");
		for (int i = 0; i < pt.length; i++)
			System.out.print("x = " + pt[i].x + "y = " + pt[i].y + " \n");

		Mat src_mat = new Mat(4, 1, CvType.CV_32FC2);
		Mat dst_mat = new Mat(4, 1, CvType.CV_32FC2);
		System.out.println((double) (pt[2].x - pt[1].x) + 20);
		System.out.println((double) (pt[3].y - pt[2].y) + 20);
		double width = 600;
		double height = 1200;
		src_mat.put(0, 0, pt[0].x, pt[0].y, pt[1].x, pt[1].y, pt[2].x, pt[2].y, pt[3].x, pt[3].y);
		dst_mat.put(0, 0, 0.0, 0.0, 0.0, height, width, height, width, 0);
		// 와핑 작업 진행
		Mat perspectiveTransform = Imgproc.getPerspectiveTransform(src_mat, dst_mat);
		Mat dst = source.clone();
		Imgproc.warpPerspective(source, dst, perspectiveTransform, new Size(width, height), Imgproc.INTER_CUBIC,
				Core.BORDER_REPLICATE, new Scalar(10));
		// 파일 작성
		Imgcodecs.imwrite(name, dst);
	}
}