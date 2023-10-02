# Decision Making Framework for the End-of-Life Buildings using Convolutional Neural Networks (CNN)

## Problem Statement

The construction industry is the main contributor to global greenhouse gas (GHG) emissions, resource
consumption and demolition-related waste generation. A reduction in these environmental impacts can be
achieved by recovering building waste through selective deconstruction and material reuse.

The built environment can take advantage of circular economy and digitalization principles to promote
reuse practices. However, the absence of an established decision-making framework to identify reusable
structural elements and the lack of initiatives to adopt advanced technologies for reuse slow down the
practical implementation in the built environment.

In this repository, a multi-criteria decision-making framework is developed with an automated convolutional
neural network (CNN) tool that recognizes corrosion, connection types and associated damage from 2D steel
structure images, with an overall accuracy of 83%, 89%, and 83%, respectively.

This app is designed to harness the convolutional neural networks and machine learning algorithms to
support the decision around the re-usability of structural elements from end-of-life buildings.

## How it works:

Input the value of different parameters to evaluate the reusability potential of steel structures from
an existing end-of-life buildings. Optionally, it is possible to upload 2D structural steel images to
evaluate the corrosion level, connection types and the associated damage on the structure. The outputs
later on merge with other parameters and the app provides a suggested response in terms of reuse/recycle.

## To launch:

1. Create a new Python environment
2. Install all dependencies from requirements.txt
```
pip install -r requirements.txt
```
3. Activate the created environment
4. Open a terminal in your code interpreter and launch the app
```
streamlit run general_decision_making_dashboard_updated.py
```

## Publication
- Kanyilmaz A., De Wolf C., Kondratenko A., Birhane M., Raghu D., (2022, March 24), Application of computer vision for the efficient
choice of end-of-life scenario of building’s structural elements [conference presentation](https://www.research-collection.ethz.ch/handle/20.500.11850/594728), AI in AEC 2022 virtual conference
