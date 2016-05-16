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

public class MainForm {
	@FXML
	private Button btnTest;
	@FXML
	private Label label;
	private ImageProcess process;
	@FXML
	private void btnTestMouseClickHandler(ActionEvent event) throws IOException {
		btnTest.setText("TEST");
		label.setText("버튼을 눌렀습니다.");
		FileChooser fileChooser = new FileChooser();
		fileChooser.setTitle("Open Resource File");
		File file = null;
		try {
			file = fileChooser.showOpenDialog(null);
			label.setText(file.getPath());
		} catch (Exception e) {
			System.out.println("Error");
		}
		process = new ImageProcess(file.getPath(), file.getName());
		process.warpingBiliardsImage(0, 1, 400, 800);		
		showImage(file.getName());
	}
	private void showImage(String path) {
		Image image = new Image("file:" + path); // 주의! file:을 붙여야 로컬로 인식한다.
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
}