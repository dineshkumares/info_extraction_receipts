info_extraction_receipts
==============================



# Introduction:

Automated Information extraction is the process of extracting structured information from unstructured/semi structured documents. This project focuses on semi-structured documents. Semi-structured documents are documents such as invoices or purchase orders that do not follow a strict format the way structured forms to, and are not bound to specified data fields. 
Information Extraction holds a lot of potential in automation. To name a few applications:
- Automatically scan images of invoices (bills) to extract valuable information.
- Build an automated system to automatically store revelant information from an invoice: eg: company name, address, date, invoice number, total  

In order to be able to do this, here are the basic steps:
- Gather raw data (invoice images)
- Optical character recognition (OCR) engine such as [Tesseract](https://tesseract-ocr.github.io) or [Google Vision](https://cloud.google.com/vision/docs/ocr).
- Extract relevant/salient information in a digestable format such as json for storage/analytics.

The main issue/concern with this approach is that invoices do not follow a universal pattern. 

<p align="center">
<img src="figures/figure_1.png" width="1000" height="500"> 

__Figure 1__: _Different patterns of semi structured documents make it difficult to generalize a pattern for Information Extraction(IE)_
