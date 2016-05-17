import javafx.animation.Interpolator;
import javafx.animation.Timeline;
import javafx.animation.TranslateTransition;
import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.geometry.VPos;
import javafx.scene.Group;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.BorderPane;
import javafx.scene.paint.Color;
import javafx.scene.shape.Rectangle;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;
import javafx.scene.text.Text;
import javafx.scene.text.TextAlignment;
import javafx.stage.Stage;
import javafx.util.Duration;

public class AppMain extends Application {
	@Override
		public void start(Stage stage) throws Exception {
			Parent root = FXMLLoader.load(getClass().getResource("MainForm.fxml"));
			Scene scene = new Scene(root);
			stage.setTitle("SmartBiliards");
			stage.setScene(scene);
			stage.sizeToScene();
			stage.show();
			
		}
	public static void main(String[] args) {
		launch(args);
		
	}
}
