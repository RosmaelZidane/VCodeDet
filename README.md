
## VCodeDet: a Graph Neural Network for Source Code Vulnerability Detection


#### VCodeDet directory Structure

```dir
(main module)  ├── templates
               │   └── images
(models)       ├── vulcodedetectmodel
               │   ├── storage
               │   |  ├── cache
               │   |  ├── processed
               │   |  ├── external
(CPG generator)|   |  |   |── joern-CLI
(pre-trained)  |   |  |   └── checkpoints
(results)      └── ── └──output     
```

### Section 1: Analyze Java code from a pre-trained VcodeDet.

1. Clone the VcodeDet repository.

```bash
git clone https://github.com/RosmaelZidane/VcodeDet.git
```
2. Clreate and activate a Python virtual environment for required packages.

```bash
python3 -m venv .myvenv
source .myvenv/bin/activate
```
3. Install the required Python packages.

```bash
pip install -r requirements.txt
```
3. Download and install the graph extractor. We use [Joern](https://joern.io/) to extract the CPG from the source code.
```bash
chmod +x ./vulcodedetectmodel/getjoern.sh
./vulcodedetectmodel/getjoern.sh
```
4. Download the model checkpoint and the required reference file, which is available at the provided [link](https://drive.google.com/drive/folders/10_MjuMhxd_hCROWWzdl7aCdSeQUToM4-?usp=sharing).
```bash
chmod +x ./vulcodedetectmodel/getreference.sh
./vulcodedetectmodel/getreference.sh
```
Check that the "cache" and "processed" folders exist in /vulcodedetectmodel/storage. Additionally, confirm that the "Vcodedet.ckpt" file is present in /vulcodedetectmodel/storage/external/checkpoints. If any of these do not exist, repeat instruction 4.

5. Almost there! To access the model via API, run the app.py file and follow the link that appears in the terminal. Once the instructions below are executed successfully, please read the details in the code input area of your browser to ensure correct graph generation for your analysis.

Next, add your code and click on [Predict]. After a few seconds, the results will display as a table below the code input area. For clarity, a new Java file named javacodeanalyse.java has been created in the project repository. Open this file to compare the results from the table with the corresponding lines of code.

```bash
python3 app.py
```

6. If you encounter any issues with this process, we recommend using the Docker version (Section 3).

### Section 2: Re-training VcodeDet from scratch on the [Project_KB](https://github.com/SAP/project-kb.git) datasets.

1. If you have already run the previous Section (1), navigate to vulcodedetectmodel/storage and delete the "cache" and "processed" folders. If not, please follow the instructions provided in Section 1 from steps 1 to 3.
```bash
cd /vulcodedetectmodel/storage && rm -rf cache/ processed/
```
2. Prepare the dataset. 
```bash
python3 ./vulcodedetectmodel/prepare_data.py
```
3. Run Joern to extract the CPG from the source code. This step may take some time depending on the system's performance.
```bash
python3 ./vulcodedetectmodel/generate_node_edge_data.py
```
4. Train and test the model.
```bash
python3 ./vulcodedetectmodel/main_model.py
```

Note: Part of the results will be displayed in the terminal. Open the /vuljavadetectmodel/storage/output directory to explore all the created CSV files for further metrics.


### Section 3: Docker Method.

It is expected that you have basic knowledge of [Docker](https://docs.docker.com/install/), which includes installation, testing, and interaction.


1. Clone the VcodeDet repository.

```bash
git clone https://github.com/RosmaelZidane/VcodeDet.git
```
2. Build a docker image
```bash
docker build -t image_Vcodedet .
```
3. Run the application on port 8000. This port should be free while running the container.
```bash
docker run -d -p 8000:8000 image_Vcodedet
```

If you cannot access the browser from the command, run the following instructions to execute the app manually.

```bash
docker run -it image_Vcodedet bash
```
Run app.y, then click on the link in the terminal.
```bash
python3 app.py
```
Exit and stop the container once you have completed the code analysis.

### Citation

To be provided.
