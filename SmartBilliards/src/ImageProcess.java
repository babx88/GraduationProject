import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import javafx.scene.image.Image;

public class ImageProcess {
	public enum Theta {
		RIGHT_TOP,
		LEFT_TOP,
		HORIZONTAL
	}
	class HoughClass { // 허프클래스를 통해 나온 결과 rho와 theta를 저장할 변수를 내부 클래스로 선언
		private double rho;
		private double theta;
		public Theta orient;
		public HoughClass(double rho, double theta) {
			this.rho = rho;
			this.theta = theta;
		}
		public HoughClass(double rho, double theta, Theta orient) {
			this.rho = rho;
			this.theta = theta;
			this.orient = orient;
		}
		@Override
		public String toString() {
			return "rho: " + rho + " theta: " + theta + " orient: " + orient + "\n"; 
		}
	}
	private Mat srcImage;
	private String fileName;
	private Mat dstImage;
	
	public ImageProcess(String path, String name) {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		srcImage = Imgcodecs.imread(path);
		dstImage = srcImage.clone();
		fileName = name;
	}
	public Mat getDstImage() {
		return dstImage;
	}
	public int[] histo() {
		Mat mask = new Mat();
		Mat hist = new Mat();
		MatOfFloat ranges=new MatOfFloat(0f, 256f);
		double[] max = new double[3];
		max[0] = max[1] = max[2] = 0;
		int[] index = new int[3];
		int k = 0;
		double[] data = new double[256];
		for(int i = 0; i < 3; i++) {
			Imgproc.calcHist(Arrays.asList(dstImage), new MatOfInt(i), mask, hist, new MatOfInt(256), ranges);
			for(int j = 0; j < 256; j++) {
				data = hist.get(j, 0);
				if(data[0] > max[k]) {
					max[k] = data[0];
					index[k] = j;
				}
			}
			k++;
		}
		System.out.println("max[0] = " + index[0] + " max[1] = " + index[1] + " max[2] = " + index[2]);
		return index;
	}
	private void removeTable() {
		int[] index = histo();
		boolean isMaxGreen = index[1] > index[2];
		double[] datas;
		for(int j = 0; j < dstImage.rows(); j++) {
			for(int i = 0; i < dstImage.cols(); i++) {
				datas = dstImage.get(j, i); // BGR 순서로 얻는다.
				if( isMaxGreen
						&& index[1] - 40 <= datas[1] && datas[1] <= index[1] + 40 ) // GREEN
					datas[0] = datas[1] = datas[2] = 0;
				else if( !isMaxGreen
						&& index[2] - 40 <= datas[0] && datas[0] <= index[2] + 40 ) // BLUE
					datas[0] = datas[1] = datas[2] = 0;
				dstImage.put(j, i, datas);
			}
		}
	}
	public void warpingBiliardsImage(int threshold1, int threshold2, double width, double height) {
		Imgproc.GaussianBlur(dstImage, dstImage, new Size(5,5), 3);
		kMeanClusters(4);
		fillArea();
		inRange(new Scalar(0,0,0), new Scalar(0,0,0), dstImage);
		
		Imgproc.dilate(dstImage, dstImage, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(16, 16)));
		Imgproc.erode(dstImage, dstImage, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(16,16))); 
		cannyEdgeDetect(threshold1, threshold1);
		ArrayList<HoughClass> hough = houghLineExtract(threshold1, threshold2);

		dstImage = srcImage.clone();
		createHoughLineOnImage(hough);
		Point[] pt = extractWarpingPoint(hough);
		warpImage(pt, width, height);
//		kMeanClusters(14);
		Mat dst = dstImage.clone();
		Point[] centroidPoint = Labelling();
		dstImage.setTo(new Scalar(85, 107, 45)); // 녹색으로 채우기.
		createBall(centroidPoint, dst);
		Imgcodecs.imwrite(fileName, dstImage);
	}
	private void createHoughLineOnImage(ArrayList<HoughClass> list) {
		double c, s, x0, y0;
		double theta, rho;
		for(int i = 0; i < list.size(); i++) {
			theta = list.get(i).theta;
			rho = list.get(i).rho;
			c = Math.cos(theta);
			s = Math.sin(theta);
			x0 = c * rho;
			y0 = s * rho;
			Point pt1 = new Point(x0 + 10000 * (-s), y0 + 10000 * c);
			Point pt2 = new Point(x0 - 10000 * (-s), y0 - 10000 * c);
			Imgproc.line(dstImage, pt1, pt2, new Scalar(0, 0, 255), 1);
			System.out.println("rho: " + rho + " // theta: " + theta);
		}
	}
	private void cannyEdgeDetect(int threshold1, int threshold2) { // 케니 엣지 추출
//		Imgproc.blur(dstImage, dstImage, new Size(3, 3)); // 영상 블러링(노이즈 제거)
		Imgproc.Canny(dstImage, dstImage, threshold1, threshold2);
	}
	private ArrayList<HoughClass> checkCorrectLine(ArrayList<HoughClass> list) {
		final int MAX = 0;
		final int MIN = 1;
		double leftRhoMean = 0, rightRhoMean = 0, horizontalRhoMean = 0;
		double leftThetaMean = 0, rightThetaMean = 0, horizontalThetaMean = 0;
		int leftRhoCount = 0, rightRhoCount = 0, horizontalRhoCount = 0;
		HoughClass leftTop = null;
		HoughClass rightTop = null;
		HoughClass[] horizontal = new HoughClass[2];
		ArrayList<HoughClass> horizontals = new ArrayList<>();
		ArrayList<HoughClass> leftTops = new ArrayList<>();
		ArrayList<HoughClass> rightTops = new ArrayList<>();
		HoughClass temp;
		for (int k = 0; k < list.size(); k++) {
			temp = list.get(k);
			if( temp.orient == Theta.LEFT_TOP ) {
				leftTops.add(temp);
				leftRhoMean += temp.rho;
				leftThetaMean += temp.theta;
				leftRhoCount++;
			}
			else if( temp.orient == Theta.RIGHT_TOP ) {
				rightTops.add(temp);
				rightRhoMean += temp.rho;
				rightThetaMean += temp.theta;
				rightRhoCount++;
			}
			else if( temp.orient == Theta.HORIZONTAL ) {
				horizontals.add(temp);
				horizontalRhoMean += temp.rho;
				horizontalThetaMean += temp.theta;
				horizontalRhoCount++;
			}
		}
		leftRhoMean /= leftRhoCount;
		rightRhoMean /= rightRhoCount;
		horizontalRhoMean /= horizontalRhoCount;
		leftThetaMean /= leftRhoCount;
		rightThetaMean /= rightRhoCount;
		horizontalThetaMean /= horizontalRhoCount;
		int maxWindowLength = dstImage.height() > dstImage.width() ? dstImage.height() : dstImage.width();
		double errorRhoRange = maxWindowLength * 0.2; // 오차범위 20%
		double errorThetaRange = Math.PI * 0.15; // 오차범위 15%
		for(int i = 0; i < leftTops.size(); i++ ) {
			temp = leftTops.get(i);
//			if( !(Math.abs(temp.rho - leftRhoMean) <= errorRhoRange) ) continue;
//			else if( !(Math.abs(temp.theta - leftThetaMean) <= errorThetaRange) ) continue;
			if(leftTop == null) leftTop = temp;
			if(temp.rho > leftTop.rho) leftTop = temp; 
		}
		for(int i = 0; i < rightTops.size(); i++ ) {
			temp = rightTops.get(i);
			if( !(Math.abs(temp.rho - rightRhoMean) <= errorRhoRange) ) continue;
//			else if( !(Math.abs(temp.theta - rightThetaMean) <= errorThetaRange) ) continue;
			if(rightTop == null) rightTop = temp;
			if(temp.rho < rightTop.rho) rightTop = temp;
		}
		for(int i = 0; i < horizontals.size(); i++ ) {
			temp = horizontals.get(i);
			if( temp.rho > horizontalRhoMean ) {
				if( !(Math.abs(temp.theta - horizontalThetaMean) <= errorThetaRange) ) continue;
				if(horizontal[MIN] == null) horizontal[MIN] = temp;	
				if(temp.rho < horizontal[MIN].rho) horizontal[MIN] = temp;
			}
			else {
				if( !(Math.abs(temp.theta - horizontalThetaMean) <= errorThetaRange) ) continue;
				if(horizontal[MAX] == null) horizontal[MAX] = temp;
				if(temp.rho > horizontal[MAX].rho) horizontal[MAX] = temp; 
			}
		}
		ArrayList<HoughClass> result = new ArrayList<>();
		result.add(leftTop); result.add(rightTop); result.add(horizontal[MAX]); result.add(horizontal[MIN]); 
		try {
			System.out.println(leftTop);
			System.out.println(rightTop);
			System.out.println(horizontal[MAX]);
			System.out.println(horizontal[MIN]);
		} catch(Exception e) {
			System.out.println("ERROR");
		}
		return list;
	}
	private boolean isCorrectLine(ArrayList<HoughClass> list, double rho, double theta) {
		boolean result = true;
		for (int k = 0; k < list.size(); k++) {
			if ( !( (0 < theta && theta < 1.1) || (2.10 < theta && theta < 2.90 ) || (1.2 < theta && theta < 2.0 ) )) {
				result = false;
				break;
			}
			
			if (Math.abs(Math.abs(list.get(k).rho) - Math.abs(rho)) < srcImage.height() * 0.1
					&& Math.abs(list.get(k).theta - theta) < Math.PI * 2 * 0.1) {
				result = false;
				break;
			}
			else if(theta < 0.3) {
				double tempTheta = Math.PI - theta;
				if( Math.abs(Math.abs(list.get(k).rho) - Math.abs(rho)) < srcImage.height() * 0.1
						&& Math.abs(list.get(k).theta - tempTheta) < Math.PI * 2 *0.1) {
					result = false;
					break;
				}
			}
		}
		return result;
	}
	private ArrayList<HoughClass> houghLineExtract(int threshold1, int threshold2) { // 허프 라인 추출
		cannyEdgeDetect(threshold2, threshold2); // 케니 엣지로 윤곽선 추출
		
		Mat lines = new Mat();
		Imgproc.HoughLines(dstImage, lines, 1, 2 * Math.PI / 180.0, 100); // 허프 완료

		double rho, theta;
		double data[];

		ArrayList<HoughClass> list = new ArrayList<>(); // 라인 자료를 저장할 ArrayList
		System.out.println(lines.rows());

		// 허프 라인 따는 부분
		for (int i = 0; i < lines.rows(); i++) { // 자바에서는 Rows와 Cols가 다르다.
			data = lines.get(i, 0);
			rho = data[0];
			theta = data[1];
			if (theta == 0)	theta = 0.00001;
			if( !isCorrectLine(list, rho, theta) ) 
				continue;
			list.add(new HoughClass(rho, theta)); 
//			if ( 0.3 < theta && theta < 1.1 ) 
//				list.add(new HoughClass(rho, theta, Theta.RIGHT_TOP)); 
//			else if( 1.2 < theta && theta < 2.0 )
//				list.add(new HoughClass(rho, theta, Theta.HORIZONTAL));
//			else if( 0 < theta && theta < 0.3 || 2.1 < theta && theta < 2.9)
//				list.add(new HoughClass(rho, theta, Theta.LEFT_TOP));
		}
		ArrayList<HoughClass> resultList = checkCorrectLine(list);
		
		// rho를 기준으로 오름차순 정렬
		java.util.Collections.sort(resultList, new Comparator<HoughClass>() {
			@Override
			public int compare(HoughClass obj1, HoughClass obj2) {
				return (Math.abs(obj1.rho) < Math.abs(obj2.rho)) ? -1
						: (Math.abs(obj1.rho) > Math.abs(obj2.rho)) ? 1 : 0;
			}
		});
		
		for (int i = 0; i < list.size(); i++)
			System.out.print("rho: " + list.get(i).rho + " " + list.get(i).orient + "\n");
		System.out.println("");
		return resultList;
	}
	private Point[] extractWarpingPoint(ArrayList<HoughClass> list) {
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
		return pt;
	}
	private void warpImage(Point[] pt, double width, double height) { // 이미지 와핑
		// 와핑 작업
		Mat src_mat = new Mat(4, 1, CvType.CV_32FC2);
		Mat dst_mat = new Mat(4, 1, CvType.CV_32FC2);
		System.out.println((double) (pt[2].x - pt[1].x) + 20);
		System.out.println((double) (pt[3].y - pt[2].y) + 20);
		src_mat.put(0, 0, pt[0].x, pt[0].y, pt[1].x, pt[1].y, pt[2].x, pt[2].y, pt[3].x, pt[3].y);
		dst_mat.put(0, 0, 0.0, 0.0, 0.0, height, width, height, width, 0);
		// 와핑 작업 진행
		Mat perspectiveTransform = Imgproc.getPerspectiveTransform(src_mat, dst_mat);
		Imgproc.warpPerspective(srcImage, dstImage, perspectiveTransform, new Size(width, height));
	}
	private void kMeanClusters(int k) {
		Mat samples = dstImage.reshape(3, dstImage.cols() * dstImage.rows());
		Mat samples32f = new Mat();
		TermCriteria criteria = new TermCriteria(TermCriteria.COUNT + TermCriteria.EPS, 10, 1.0);
		samples.convertTo(samples32f, CvType.CV_32F, 1.0 / 255.0);
		
		Mat labels = new Mat();
		Mat centers = new Mat();
		Core.kmeans(samples32f, k, labels, criteria, 1, Core.KMEANS_PP_CENTERS, centers);

		centers.convertTo(centers, CvType.CV_8UC1, 255.0);
		centers.reshape(3);

		int rows = 0;
		for (int y = 0; y < dstImage.rows(); y++) {
			for (int x = 0; x < dstImage.cols(); x++) {
				int label = (int) labels.get(rows, 0)[0];
				int r = (int) centers.get(label, 2)[0];
				int g = (int) centers.get(label, 1)[0];
				int b = (int) centers.get(label, 0)[0];
				dstImage.put(y, x, b, g, r);
				rows++;
			}
		}
	}
	private void inRange(Scalar lower, Scalar upper, Mat dst) {
		Mat hsvImage = new Mat();
		Imgproc.cvtColor(dst, hsvImage, Imgproc.COLOR_BGR2HSV);
		Scalar lowerB = lower;
		Scalar upperB = upper;
		Core.inRange(hsvImage, lowerB, upperB, dst);
	}
	private void fillArea() {
		Mat mask = new Mat();
		Imgproc.floodFill(dstImage, mask, new Point((int)(dstImage.width()/ 2), (int)(dstImage.height() / 2)), new Scalar(0, 0,0));
		Imgproc.floodFill(dstImage, mask, new Point((int)(dstImage.width()/ 2 + 100), (int)(dstImage.height() / 2) + 100), new Scalar(0,0,0));
		Imgproc.floodFill(dstImage, mask, new Point((int)(dstImage.width()/ 2 - 100), (int)(dstImage.height()/ 2) - 100 ), new Scalar(0,0,0));
//		Imgproc.floodFill(dstImage, mask, new Point((int)(dstImage.width()/ 2), (int)(dstImage.height() / 2)), new Scalar(0,0, 255)
//				, new Rect(), new Scalar(10), new Scalar(10), 8 | Imgproc.FLOODFILL_MASK_ONLY);
	}
	private void houghCirclesExtract() {
		Mat circles = new Mat();
		int minRadius = 8;
		int maxRadius = 60;
		Mat dst = dstImage.clone();
		Imgproc.cvtColor(dst, dst, Imgproc.COLOR_BGR2GRAY);
//		Imgproc.GaussianBlur(dst, dst, new Size(3,3), 2);
		double data[];
		int cx, cy, r;
		Imgproc.HoughCircles(dst, circles, Imgproc.CV_HOUGH_GRADIENT, 1, 20, 30, 20, minRadius, maxRadius);
		System.out.println(circles.cols() + "입니다.");
		for(int i = 0; i < circles.cols(); i++) {
			data = circles.get(0, i);
			cx = (int) Math.round(data[0]);
			cy = (int) Math.round(data[1]);
			r = (int) Math.round(data[2]);
			
			Point center = new Point(cx, cy);
			Imgproc.circle(dstImage, center, r, new Scalar(0, 0, 255), 2);
		}
	}
	private Point[] Labelling() {
		Mat label = new Mat();
		Mat stats = new Mat();
		Mat centroids = new Mat();
		Mat dst = dstImage.clone();
//		inRange(new Scalar(0, 60, 20), new Scalar(140, 180, 150), dstR);	// RED
//		inRange(new Scalar(20, 100, 100), new Scalar(30, 255, 255), dstY);	// YELLOW
//		inRange(new Scalar(0, 0, 240), new Scalar(170, 20, 255), dstW);	// WHITE
		double[] datas;
		for(int j = 0; j < dst.rows(); j++) {
			for(int i = 0; i < dst.cols(); i++) {
				datas = dst.get(j, i); // BGR 순서로 얻는다.
				if( datas[0] > 200 && datas[1] > 200 && datas[2] > 200 )  // W
					datas[0] = datas[1] = datas[2] = 255;
				else if( datas[0] < 115 && datas[1] < 115 && datas[2] > 150 ) // RED
					datas[0] = datas[1] = datas[2] = 255;
				else if( datas[1] > 160 && datas[2] > 180 )  // YELLOW
					datas[0] = datas[1] = datas[2] = 255;
				else datas[0] = datas[1] = datas[2] = 0;
				dst.put(j, i, datas);
			}
		}
		Imgproc.erode(dst, dst, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3,3)));  
		Imgproc.dilate(dst, dst, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(8, 8)));
		Imgproc.erode(dst, dst, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(12, 12)));  
		Imgproc.dilate(dst, dst, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(4, 4)));
	
//		dstImage = dst;
		Imgproc.cvtColor(dst, dst, Imgproc.COLOR_BGR2GRAY);
		Imgproc.connectedComponentsWithStats(dst, label, stats, centroids, 8, CvType.CV_32S);
		System.out.println("라벨 수 : " + centroids.rows());
		Point[] centroidPoint = new Point[ centroids.rows() ];
		
		double[] data = new double[2];
		for (int i = 1; i < centroids.rows(); i++) {
			Mat region = centroids.row(i);
			region.get(0, 0, data);
			centroidPoint[i] = new Point(data[0], data[1]);
			System.out.println("x: " + centroidPoint[i].x + " y: " + centroidPoint[i].y);
		}
		return centroidPoint;
	}
	private void createBall(Point[] centroidPoint, Mat dst) {
		double[] data;
		Scalar color = null;
		for(int i = 1; i < centroidPoint.length; i++) {
			data = dst.get((int)centroidPoint[i].y, (int)centroidPoint[i].x);
			if( data[0] > 200 && data[1] > 200 && data[2] > 200 )  // WHITE
				color = new Scalar(255, 255, 255);
			else if( data[0] < 115 && data[1] < 115 && data[2] > 150 ) // RED
				color = new Scalar(0, 0, 255);
			else if( data[1] > 160 && data[2] > 180 )  // YELLOW
				color = new Scalar(0, 255, 255);
			else color = new Scalar(0, 0, 255);
			Imgproc.circle(dstImage, centroidPoint[i], 5, color, 12);			
		}	
	}
}
