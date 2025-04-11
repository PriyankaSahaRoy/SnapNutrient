#import_necessary_library_file
import cv2
import pandas as pd
from ultralytics import YOLO
import matplotlib.pyplot as plt

#load_the_YOLOV8N_Pretrained_model
model = YOLO("yolov8n.pt")

#load_nutrition_dataset_as_csv_file
nutrition_df = pd.read_csv("coco_food_nutrition.csv")
nutrition_df['Food'] = nutrition_df['Food'].str.lower()

#Nutrient_labels_to_display
nutrient_labels = ['Calories (kcal)', 'Protein (g)', 'Fat (g)', 'Carbohydrates (g)', 'Fiber (g)', 'Sugars (g)',
                   'Calcium (mg)', 'Iron (mg)', 'Magnesium (mg)', 'Phosphorus (mg)', 'Potassium (mg)',
                   'Sodium (mg)', 'Zinc (mg)', 'Vitamin C (mg)', 'Cholesterol (mg)']

#Function_to_annotate_frames
def annotate_frame(frame, results, detected_foods):  #Defines_a_function_to_draw_detection_boxes_and_nutrition_info_on_the_image.
    font = cv2.FONT_HERSHEY_SIMPLEX #Sets_a_simple_sans-serif_font_style_for_text_display.
    font_color = (139, 0, 0) #Defines_the_text_color_as_dark_blue_in_BGR_format.
    font_scale = 0.5   #Sets_the_text_size_to_half_of_the_default_size.

    for result in results: #Iterates over detection results returned by the YOLO model.
        for box in result.boxes:  #Loops through each detected bounding box in the result.
            cls_id = int(box.cls[0].item())  #Extracts the predicted class ID of the object.
            class_name = model.names[cls_id].lower() #Converts the class ID to the corresponding class name (e.g., 'banana') in lowercase.

            if class_name in nutrition_df['Food'].values:  #Checks if the detected food is listed in the nutrition data CSV.
                detected_foods.add(class_name)  #Adds the food item to a set of detected food names to avoid duplication.
                conf = float(box.conf[0])
                coords = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = coords

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 100, 200), 2)
                label = f"{class_name.title()} ({conf:.2f})"
                cv2.putText(frame, label, (x1, y1 - 10), font, font_scale, font_color, 1, cv2.LINE_AA)

                nut_row = nutrition_df[nutrition_df['Food'] == class_name]
                if not nut_row.empty:
                    y_offset = y1 + 20
                    for nut_label in nutrient_labels[:5]:
                        val = nut_row.iloc[0][nut_label]
                        nut_text = f"{nut_label}: {val}"
                        cv2.putText(frame, nut_text, (x1, y_offset), font, font_scale, font_color, 1, cv2.LINE_AA)
                        y_offset += 18
    return frame

#option_for_detection_mode
mode = input("Select detection mode:\n1. Static Image\n2. Live Camera\nEnter 1 or 2: ")

detected_foods_total = set()

if mode == '1':
    image_path = input("Enter the image path (e.g., test.jpg): ")
    image = cv2.imread(image_path)

    if image is None:
        print("Could not load the image. Check the path.")
        exit()

    results = model(image)
    annotated = annotate_frame(image.copy(), results, detected_foods_total)

    #Convert_BGR_to_RGB_for_matplotlib
    image_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    #Plot_image_and_nutrition_summary
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))

    axs[0].imshow(image_rgb)
    axs[0].set_title("Detected Foods with Nutrition Info")
    axs[0].axis("off")

    if detected_foods_total:
        nutrients_combined = []
        labels_combined = []
        for food in detected_foods_total:
            food_row = nutrition_df[nutrition_df['Food'] == food]
            if not food_row.empty:
                nutrients = food_row[nutrient_labels].values.flatten().astype(float)
                labels = [f"{food.title()} - {label}" for label in nutrient_labels]
                nutrients_combined.extend(nutrients)
                labels_combined.extend(labels)

        axs[1].bar(labels_combined, nutrients_combined)
        axs[1].set_title("Nutrient Summary")
        axs[1].set_ylabel("Amount per 100g")
        axs[1].tick_params(axis='x', rotation=90)
    else:
        axs[1].text(0.5, 0.5, "No food items detected.", ha='center', va='center')
        axs[1].axis("off")

    plt.tight_layout()
    plt.show()

elif mode == '2':
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not access the camera.")
        exit()

    print("Live food detection started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame not received from camera.")
            break

        resized = cv2.resize(frame, (640, 480))
        results = model(resized, verbose=False)
        frame_annotated = annotate_frame(resized.copy(), results, detected_foods_total)

        cv2.imshow("Live Food Detection", frame_annotated)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    #Show_nutrient_bar_chart_after_live_session
    if detected_foods_total:
        print("\nNutrient Summary for Detected Foods:")
        plt.figure(figsize=(12, 6))
        for food in detected_foods_total:
            food_row = nutrition_df[nutrition_df['Food'] == food]
            if not food_row.empty:
                print(f"\n{food.title()}:")
                for nut in nutrient_labels:
                    print(f"  {nut}: {food_row.iloc[0][nut]}")
                nutrients = food_row[nutrient_labels].values.flatten().astype(float)
                plt.bar([f"{food.title()} - {label}" for label in nutrient_labels], nutrients)

        plt.xticks(rotation=90)
        plt.ylabel("Amount per 100g")
        plt.title("Detected Food Nutrient Summary")
        plt.tight_layout()
        plt.show()
    else:
        print("No food items detected.")
else:
    print("Invalid option. Please enter 1 or 2.")
