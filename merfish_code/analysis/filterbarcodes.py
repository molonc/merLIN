
from merfish_code.core import analysistask
from merfish_code.util import barcodedb

class FilterBarcodes(analysistask.AnalysisTask):

    '''An analysis task that filters barcodes based on area and mean 
    intensity.
    '''

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        self.areaThreshold = 3
        self.intensityThreshold = 200

    def get_barcode_database(self):
        return barcodedb.BarcodeDB(self.dataSet, self)

    def get_estimated_memory(self):
        return 1000

    def get_estimated_time(self):
        return 15

    def run_analysis(self):
        self.decodeTask = self.dataSet.load_analysis_task(
                self.parameters['decode_task'])        

        barcodeDB = self.get_barcode_database()
        for currentBC in self.decodeTask.get_barcode_database() \
                .get_filtered_barcodes(
                    self.areaThreshold, self.intensityThreshold, 
                    chunksize=10000):
            barcodeDB.write_barcodes(currentBC)
