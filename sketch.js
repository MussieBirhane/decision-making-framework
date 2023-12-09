const Status = (performance, threshold) => (performance >= threshold ? "Passed" : "Not passed");

let Load_Button;
let images = [];
function setup() {
  describe("Web application for CNN Tool");

  //Load_Button = createFileInput(handleImage, true).parent("#Load_Button");

  //images.forEach((img, index) => {
  //document.getElementById("#Sample_Images").innerHTML = "aa"; //image(img, 0, y, 50, 50);
  //});

  //   Load_Button = document.getElementById("Load_Button");
  //Load_Button.addEventListener("click", handleFiles, true);

  //-------------------------------------------------

  let Logistic_Performance =
    3 * (1 - parseInt(Item_Weight.value) / 4) +
    3 * parseInt(Easy_Handle.value) +
    4 * parseInt(Exist_Infrastructure.value) +
    1 * parseInt(Special_Protection.value) +
    3 * parseInt(Dismantle_Phase.value) +
    3 * parseInt(Storage_Availability.value);

  //-------------------------------------------------

  let N_Bolted_Images = 0;
  let N_Corroded_Images = 20;
  let N_Damaged_Images = 0;
  let N_Images = N_Corroded_Images + N_Bolted_Images + N_Damaged_Images;

  document.getElementById("N_Corroded_Images").innerHTML += " " + round((N_Corroded_Images / N_Images) * 100, 1) + "% of the images is corroded.";

  let Image_Classification_Performance =
    3 * (N_Bolted_Images / N_Images) +
    4 * (N_Corroded_Images / N_Images) +
    4 * (N_Damaged_Images / N_Images) +
    1 * parseInt(Composite_Connection.value) +
    1 * parseInt(Fire_Protection.value) +
    2 * parseInt(Sufficient_Amount.value) +
    2 * parseInt(Geometry_Check.value);

  //-------------------------------------------------

  let Structural_Performance =
    4 * (parseInt(Data_Quality.value) / 4) +
    2 * parseInt(Construction_Period.value) +
    3 * parseInt(Maintenance.value) +
    3 * parseInt(Purpose.value) +
    3 * parseInt(Testing.value);

  //-------------------------------------------------

  //"According to the cradle-to-cradle life cycle assessment, the embodied carbon at different stage is reported below:"
  //"Total weight of structural element: " + total_weight + " kg"
  //"Product stage A1-A3: " + total_weight * 1.13 + " kgCO2e"

  //"End of life stage C1-C4: " + total_weight * 0.018 + " kgCO2e";
  //"Reuse, recycle and recovery stage D: " + total_weight * -0.413 + " kgCO2e";

  //-------------------------------------------------

  let Logistic_Performance_Percentage = (Logistic_Performance / 17) * 100;
  let Logistic_Performance_Status = Status(Logistic_Performance_Percentage, 75);

  let Image_Classification_Performance_Percentage = (Image_Classification_Performance / 17) * 100;
  let Image_Classification_Performance_Status = Status(Image_Classification_Performance_Percentage, 70);

  let Structural_Performance_Percentage = (Structural_Performance / 15) * 100;
  let Structural_Performance_Status = Status(Structural_Performance_Percentage, 80);

  document.getElementById("Logistic").innerHTML =
    "1) Logistic feasibility: " + round(Logistic_Performance_Percentage, 2) + "% --> " + Logistic_Performance_Status;

  document.getElementById("Inspection").innerHTML =
    "2) Structural visual inspection: " + round(Image_Classification_Performance_Percentage, 2) + "% --> " + Image_Classification_Performance_Status;

  document.getElementById("Performance").innerHTML =
    "3) Structural performance: " + round(Structural_Performance_Percentage, 2) + "% -->  " + Structural_Performance_Status;

  if (Image_Classification_Performance_Status == "Passed" && Logistic_Performance_Status == "Passed" && Structural_Performance_Status == "Passed") {
    document.getElementById("Result").innerHTML += " Dismantle - Reuse";
    console.log(
      "The overall reusability analysis shows " +
        ((Structural_Performance / 3 + Image_Classification_Performance / 5 + Logistic_Performance / 3) * 100) / 3 +
        "%"
    );
  } else if (Image_Classification_Performance_Status == NaN || Logistic_Performance_Status == NaN || Structural_Performance_Status == NaN) {
    document.getElementById("Result").innerHTML = "Expecting Inputs...";
  } else {
    document.getElementById("Result").innerHTML += " Demolition - Recycle";
  }
}

function draw() {
  noLoop();
}

function Predict_Image(img, model) {
  /*   //Convert to a batch of 1
  let xb = to_device(img.unsqueeze(0), device);
  let xb = img.unsqueeze(0);
  // Get predictions from model
  let yb = model(xb);
  //Pick index with highest probability
  _, (preds = torch.max(yb, (dim = 1)));
  //Retrieve the class label */
  return null;
}

function handleImage(file) {
  if (file.type === "image") {
    img = createImg(file.data, "");
    img.hide();
  } else {
    img = null;
  }
}
