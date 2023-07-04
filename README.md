# सतर्क AI - Leveraging AI for Effective Natural Disaster Management
<p align="center">
  <img src="https://github.com/deeptiii13/Girl-Hackathon-23/assets/103764966/27430296-9b76-471e-948b-aee270c02024" width="800" height="400">
</p>
<!-- ![image](https://github.com/deeptiii13/Girl-Hackathon-23/assets/103764966/27430296-9b76-471e-948b-aee270c02024 | =100x100) -->

Natural disasters are formidable and profoundly destructive events resulting from natural processes. They encompass a range of cataclysmic occurrences, such as earthquakes, hurricanes, floods, wildfires, and tsunamis. When unleashed, these calamities cause widespread devastation of unprecedented magnitude. Entire communities can be leveled, infrastructure destroyed, and lives lost. Dwellings are reduced to rubble, crops and livelihoods are decimated, and vital services, such as electricity, clean water, and transportation, are severely disrupted. The economic toll can be staggering, with substantial financial losses incurred. The extensive destruction wrought by natural disasters serves as a poignant reminder of the immense power and inherent unpredictability of nature. It underscores the imperative for robust disaster preparedness, effective mitigation strategies, and resilient infrastructure to safeguard lives, expedite recovery, and minimize the human and societal toll.

Natural disaster prediction is crucial for early warnings, accurate risk assessment, and data-driven decision making. Artificial Intelligence helps to analyse complex data, identify patterns, and provide insights for proactive measures, resource allocation, and long-term planning. 

The early warning systems enhance disaster preparedness and facilitate effective response strategies. AI excels at identifying complex patterns in the data and adapting to changing environmental conditions. The training on historical data enables recognition of precursors of disasters and implementation of proactive measures such as reinforcing infrastructure, EWS, etc.

AI-powered systems provide valuable insights to Govt. agencies for facilitating response strategies, resource allocation and evacuation plans. They reduce the potential risks on the emergency responders by encompassing global positioning system, remote sensing, object detection etc. They benefit the public by providing timely warnings about the disasters.

# Disaster Management Cycle

1. **Preparedness**: This stage involves activities undertaken in advance to enhance the ability to respond to disasters. It includes developing emergency plans, conducting risk assessments, establishing early warning systems, training personnel, and educating communities about disaster risks and response procedures.

2. **Mitigation**: Mitigation refers to actions taken to reduce or eliminate the risks and vulnerabilities associated with disasters. This stage includes implementing measures such as constructing disaster-resistant infrastructure, implementing land-use planning strategies, enforcing building codes, and implementing environmental conservation practices.

3. **Response**: The response stage involves immediate actions taken during and immediately after a disaster to save lives, alleviate suffering, and meet the basic needs of affected populations. It includes activities such as search and rescue operations, emergency medical assistance, distribution of relief supplies, setting up temporary shelters, and restoring critical services like water and electricity.

4. **Recovery**: The recovery stage focuses on restoring the affected area to a state of normalcy and rebuilding communities. It involves efforts to repair or reconstruct damaged infrastructure, support livelihoods, provide psychosocial support, and assist in the restoration of essential services. Long-term recovery may also include economic revitalization, community development, and efforts to reduce future disaster risks.

5. **Prevention**: Prevention involves measures taken to prevent or minimize the occurrence of future disasters. It includes ongoing efforts to improve early warning systems, enhance preparedness capacities, conduct research on disaster risks, and promote policies and practices that reduce vulnerabilities and promote resilience.

# Idea
- The development of new components involves designing novel algorithms, creating data preprocessing techniques, developing predictive models, or building user interfaces for data visualization and decision-making.
- The selection of technologies was dependent on the project's objectives, requirements, and available resources.
- Scaling parameters for a natural disaster prediction system include factors such as the size and complexity of the dataset, the frequency of data updates, the number of users or systems accessing the predictions, and the required processing capacity.
- The rollout strategy for a natural disaster prediction system involves a phased approach. It includes data collection and preprocessing, algorithm development and training, validation and testing, integration with existing systems, pilot deployments, and gradual expansion to wider user or geographical areas.
- Privacy and security considerations are crucial in natural disaster prediction systems, as they involve handling sensitive data. It is important to address these concerns by safeguarding data confidentiality, establishing secure storage and transmission protocols, and adhering to applicable data protection regulations. 

# Technologies Used 
- Programming Language (Python)
- Machine Learning Algorithms (Random Forest Classifier, GuassianNB, Prophet, YOLO)
- Natural Language Processing (NLTK)
- Frameworks (Tensorflow)
- Virtual Environment (Google Colaboratory)

# Impact

The proposed project addresses a significant societal challenge by enhancing the ability to predict and mitigate natural disasters. 

The application is grounded in extensive research and data analysis, incorporating historical data, environmental factors, patterns associated with past disasters and outliers due to destruction of nature caused by humans. 

The developed AI models will be integrated with existing monitoring systems to ensure continuous flow of information. The pre-requisite infrastructure such as computing resources, cloud storage,etc., will be established to support the deployment. The establishment of collaboration with Govt. and relevant stakeholders in order to ensure data flow and privacy. It will be followed by robust testing and validation. Furthermore, a user-friendly interface will be developed for the end-users. 

The expected outcomes of the project are multifold. It anticipates a substantial reduction in the loss of lives and property by providing timely warnings and enabling proactive measures. It aims to enhance the preparedness and response capabilities of government agencies and emergency responders through accurate predictions. It contributes to the advancement of scientific research and knowledge by comprehensively analyzing and interpreting intricate data patterns linked to natural disasters.

# Feasibility

A well-developed and practical plan has been carved out to execute the proposal. Having heard stories about how negligence may increase the impact fourfold from my dad (Fire Safety, AAI), it has always been intriguing to curb the destruction caused by these disasters and lack of preparedness.

In order to lay the foundation, I researched and identified the publicly available and meaningful datasets to train the AI model. Historical data pertaining to meteorological, geological, and satellite information has been accessed. Preprocessing has been performed to ensure the data quality, accuracy and reliability of the AI model. 

The experience in research provided an easier start to approach the problem statement. However, the lack of sufficient and authentic data posed a major problem. By leveraging the expertise of professionals, relevant data can be acquired, robust models can be developed and higher accuracy can be achieved.

# AI Usage

1. **Prediction of Earthquakes**: The proposal employs AI algorithms and models to meticulously analyze seismic data, historical patterns, and pertinent factors in order to facilitate earthquake prediction. AI amplifies the effectiveness of early warning systems, enabling proactive measures to mitigate the impact of such events.
<p align = "center">
<img alt="Earthquakes" src="https://github.com/deeptiii13/Girl-Hackathon-23/assets/103764966/3ef8bb2b-50d1-4f3b-abc0-2f0cad402861" width="800" height="500">
</p>

**Data** 

[IndianEarthquakeData.csv](https://www.kaggle.com/datasets/parulpandey/indian-earthquakes-dataset2018-onwards?resource=download): The National Center for Seismology is the nodal agency of the Government of India for monitoring earthquake activity in the country. NCS maintains the National Seismological Network of 115 stations each having state of art equipment and spreading all across the country.NCS monitors earthquake activity all across the country through its 24x7 round-the-clock monitoring center. NCS also monitors earthquake swarm and aftershock through deploying a temporary observatory close to the affected region. This dataset includes a record of the date, time, location, depth, magnitude, and source of every Indian earthquake since 2018.

**Model**

PROPHET: Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality. It works best with time series that have strong seasonal effects and several seasons of historical data. Prophet is robust to missing data and shifts in the trend, and typically handles outliers well.

**Dependencies**

```
pip install numpy
pip install pandas
pip install matplotlib
pip install prophet
pip install folium
pip install seaborn
```

   
2. **Prediction of Extreme Weather Conditions**: The proposal integrates AI technology to examine environmental data encompassing weather patterns, atmospheric conditions, and historical records. It strives to identify discernible patterns that serve as early signals for the probability of extreme weather events like floods, thunderstorms, and cyclones. It empowers the provision of early warnings and facilitates improved preparedness measures to mitigate the potential impact.
<p align="center">
  <img alt="Weather" src="https://github.com/deeptiii13/Girl-Hackathon-23/assets/103764966/3df74c55-7e11-4b57-b6c7-986322d1ee27" width="800" height="500">
</p> 

**Data** 

[WeatherData.csv](https://www.kaggle.com/datasets/muthuj7/weather-dataset?datasetId=6087): The dataset includes a record of the date, summary, type of precipitation, temperature, humidity, wind speed, visibility, cloud cover, etc. It allows the identification of patterns followed by natural disasters.

**Model**

Gaussian Naive Bayes: It is the easiest and one of the most rapid classification methods available, and it is well suited for dealing with enormous amounts of information. It makes predictions about unknown classes using the Bayes theory of probability.

Random Forest Classifier: A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.


**Dependencies**

```
pip install numpy
pip install pandas
pip install matplotlib
pip install -U scikit-learn
pip install seaborn
pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.12
```

**Usecase**

> Heavy rainfall and dangerously strong winds can unleash a cascade of devastating natural disasters. It stirs emotions by evoking fear, loss, and resilience in the face of floods, cyclones, and thunderstorms.
> High humidity and temperature can contribute to the conditions that increase the likelihood or severity of certain types of natural disasters.



1.  Floods: Heavy rainfall over an extended period or intense rainfall within a short duration causes the soil to become saturated. Excess water accumulates on the surface, leading to overflowing rivers, lakes, and drainage systems. This excessive runoff causes flooding, damaging infrastructure, disrupting transportation, and endangering lives.


2.  Cyclones: Warm ocean waters provide the energy for cyclones to form and intensify. When the cyclone travels across the ocean, the associated moisture content increses. It is released as heavy rainfall when the cyclone makes landfall. The combination of strong winds and heavy rainfall may result in severe flooding and storm surges.


3. Thunderstorms: Thunderstorms often result from unstable atmospheric conditions. When the moist air rises and cools, it forms clouds producing heavy rainfall. Intense rainfall within thunderstorms can lead to localized flash flooding, as the ground may be unable to absorb the rapid influx of water.


4. Heatwaves: High humidity, combined with high temperatures, can exacerbate the impacts of heatwaves. When humidity is high, the body's ability to cool itself through sweating is reduced, leading to increased discomfort and heat-related illnesses. Prolonged exposure to extreme heat and humidity can be dangerous.

![image](https://github.com/deeptiii13/Girl-Hackathon-23/assets/103764966/3e7178f9-272c-451e-b4f8-6eb9f0bb2512)

  
3. **Natural Language Processing for Tweets**: The proposal leverages natural language processing (NLP) techniques to analyze tweets and social media data related to disasters. Through the utilization of AI algorithms, the model can extract valuable information, identify sentiments, and detect relevant keywords to assess the severity, impact, and location of the disaster in real-time.
<p align="center">
  <img alt="NLP" src="https://github.com/deeptiii13/Girl-Hackathon-23/assets/103764966/46b3e986-86b1-4eec-a59d-e7d4afce9a72" width="400" height="800">
</p>

**Data** 

[tweets.csv](https://www.kaggle.com/datasets/vstepanenko/disaster-tweets?datasetId=519753): The file contains over 11,000 tweets associated with disaster keywords like “crash”, “quarantine”, and “bush fires” as well as the location and keyword itself. The data structure was inherited from Disasters on social media. The tweets were collected on Jan 14th, 2020.

**Model**

Tensorflow SGD (Gradient descent optimizer): Optimizers are the expanded class, which includes the method to train your machine/deep learning model. Right optimizers are necessary for your model as they improve training speed and performance. It performs redundant computations for bigger datasets, as it recomputes gradients for the same example before each parameter update. It performs frequent updates with a high variance that cause the objective function to fluctuate heavily.


**Dependencies**

```
pip install numpy
pip install pandas
pip install matplotlib
pip install -U scikit-learn
pip install seaborn
pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.12
pip install --user -U nltk
```

4. **Aerial Object Detection**: It employs AI-based computer vision algorithms to analyze aerial imagery for the detection and identification of hazards, such as damaged infrastructure, blocked roads, or stranded individuals, to assist in disaster assessment and response planning.
<p align="center">
<img width="320" alt="Screenshot 2023-07-04 at 6 52 24 AM" src="https://github.com/deeptiii13/Girl-Hackathon-23/assets/103764966/608e958a-f4bd-4e67-be8d-c00e5b4d6a4e">
</p>

**Data** 

[Aerial Images Object Detection (Ortophotos of São Paulo city in 2017)](https://www.kaggle.com/datasets/andasampa/ortofotos-2017-rgb): The dataset was developed in 2017 in São Paulo. It allows the aerial detection of objects in the image.

**Model**

YOLOv5: YOLOv5 is an advanced object detection model that builds upon the success of previous YOLO (You Only Look Once) versions. It is designed to accurately and efficiently detect objects in real-time applications. YOLOv5 introduces various improvements, including a streamlined architecture, a focus on model size reduction, and enhanced performance. It utilizes deep convolutional neural networks to analyze input images, predict bounding boxes, and classify objects within those boxes. With its efficient design and improved accuracy, YOLO v5 has become a popular choice for object detection tasks in computer vision applications.


**Dependencies**

```
pip install numpy
pip install pandas
pip install matplotlib
pip install -U scikit-learn
pip install seaborn
pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.12
%cd yolov5
%pip install -qr requirements.txt
pip install -q kaggle
pip install rasterio
```



# References

- https://www.sciencedirect.com/science/article/abs/pii/S1367912012001538
- https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=7d2f2f972cd327543d1ecc67c6afbd9dab106042
- https://nopr.niscpr.res.in/handle/123456789/11079
- https://www.researchgate.net/profile/Aakash-Parmar-2/publication/319503839_Machine_Learning_Techniques_For_Rainfall_Prediction_A_Review/links/59afb922458515150e4cc2e4/Machine-Learning-Techniques-For-Rainfall-Prediction-A-Review.pdf
- https://arxiv.org/abs/1605.05894


