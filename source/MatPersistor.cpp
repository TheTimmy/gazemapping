#include "MatPersistor.h"
#include <iostream>

using namespace std;

MatPersistor::MatPersistor(const string &fileName) {
    _pFile = NULL;
    _fileName = fileName;
    _mode = -1;
    _currentRow = -1;
}

MatPersistor::~MatPersistor() {
    close();
}

bool MatPersistor::isOpen() {
    return (_pFile != NULL);
}

bool MatPersistor::exists() {
    FILE *file = fopen(_fileName.c_str(), "r");
    if (file != NULL) {
        fclose(file);
        return true;
    }
    return false;
}

void MatPersistor::close() {
    if (_pFile != NULL) {
        if (_mode == WRITE) {
            writeHeader();
        }

        fclose(_pFile);
        _pFile = NULL;
        _mode = -1;
    }
}

bool MatPersistor::openRead() {
    return open(READ);
}

bool MatPersistor::open(int mode) {
    assert(!isOpen());

    _mode = mode;
    if (mode == READ) {
        _pFile = fopen(_fileName.c_str(), "rb");
    } else {
        _pFile = fopen(_fileName.c_str(), "rb+");
    }

    if (_pFile == 0) {
        std::cout << "Can not open: " << _fileName << std::endl;
        close();
        return false;
    }

    if (!readHeader()) {
        close();
        return false;
    }

    _currentRow = 0;
    return true;
}


bool MatPersistor::create(int cols, int type) {
    Mat aux(0, cols, type);
    return create(aux);
}


bool MatPersistor::create(Mat &mat) {
    if (isOpen()) {
        close();
    }

    _mode = WRITE;
    _pFile = fopen(_fileName.c_str(), "wb");
    if (_pFile == 0) {
        close();
        return false;
    }

    _header.rows = 0;
    _header.cols = mat.cols;
    _header.type = mat.type();

    if (!writeHeader()) {
        close();
        return false;
    }

    append(mat);

    close();
    return true;
}


bool MatPersistor::openWrite() {
    return open(WRITE);
}

bool MatPersistor::readHeader() {
    size_t hdr_size = sizeof(Header);

    fseek(_pFile, 0, SEEK_SET);
    if (fread(static_cast<void*>(&_header), 1, hdr_size, _pFile) != hdr_size) {
        return false;
    }

    return true;
}

bool MatPersistor::writeHeader() {
    size_t hdr_size = sizeof(Header);

    fseek(_pFile, 0, SEEK_SET);
    if (fwrite(static_cast<void*>(&_header), 1, hdr_size, _pFile) != hdr_size) {
        return false;
    }

    return true;
}

void MatPersistor::append(const Mat &mat) {
    append(mat, mat.rows);
}

void MatPersistor::append(const Mat &mat, int rows) {
    assert(isOpen() && _mode == WRITE);
    assert(_header.cols == mat.cols && _header.type == mat.type());
    assert(mat.rows >= rows);

    long bytes = rows * mat.cols * mat.elemSize();
    fseek(_pFile, 0, SEEK_END);
    long written = fwrite((char *) mat.data, 1, bytes, _pFile);
    assert (bytes == written);

    _header.rows += rows;
    _currentRow += rows;
}

void MatPersistor::read(Mat &mat) {
    read(mat, _header.rows);
}

int MatPersistor::read(Mat &mat, int maxRows) {
    assert(isOpen() && _mode == READ);

    int toRead = min(_header.rows - _currentRow, maxRows);
    if (mat.cols == 0 || toRead > mat.rows ||
        _header.cols != mat.cols ||
        _header.type != mat.type()) {

        mat.create(toRead, _header.cols, _header.type);
    }

    long bytes = toRead * mat.cols * mat.elemSize();
    long read = fread((char *)mat.data, 1, bytes, _pFile);
    assert (bytes == read);

    _currentRow += toRead;
    return toRead;
}

int MatPersistor::cols() {
    assert(isOpen());
    return _header.cols;
}

int MatPersistor::rows() {
    assert(isOpen());
    return _header.rows;
}

int MatPersistor::type() {
    assert(isOpen());
    return _header.type;
}

int MatPersistor::elementSize() {
    assert(isOpen());
    Mat dummy(1, 1, _header.type);
    return dummy.elemSize();
}

void MatPersistor::setRow(int row) {
    assert(isOpen());

    if (_currentRow == row) {
        return;
    }

    assert(0 <= row && row < _header.rows);

    int hdrSize = sizeof(Header);
    int rowSize = cols() * elementSize();
    long offset = hdrSize + (row * rowSize);
    fseek(_pFile, offset, SEEK_SET);

    _currentRow = row;
}