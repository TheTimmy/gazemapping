#include "GazePointPersistor.h"

#include <fstream>
#include <sstream>
#include <iostream>

/*
 Thanks to @Martin York for the awesome implementation of a csv reader.
 https://stackoverflow.com/questions/1120140/how-can-i-read-and-parse-csv-files-in-c
*/
class CSVRow {
    public:
        std::string operator[](std::size_t index) const {
            return std::string(&m_line[m_data[index] + 1], m_data[index + 1] - (m_data[index] + 1));
        }

        operator bool () const {
            return size() > 0;
        }

        std::size_t size() const {
            return m_data.size() - 1;
        }
        
        void readNextRow(std::istream& str) {
            std::getline(str, m_line);

            m_data.clear();
            m_data.emplace_back(-1);
            std::string::size_type pos = 0;
            while((pos = m_line.find('\t', pos)) != std::string::npos) {
                m_data.emplace_back(pos);
                ++pos;
            }
            // This checks for a trailing comma with no data after it.
            pos   = m_line.size();
            m_data.emplace_back(pos);
        }

    private:
        std::string         m_line;
        std::vector<int>    m_data;
};

std::istream& operator>>(std::istream& str, CSVRow& data) {
    data.readNextRow(str);
    return str;
}

GazePointPersistor::GazePointPersistor(GazePointPersistor&& o)
    : gazes(std::move(o.gazes)), framerate(o.framerate), converter(o.converter), is_open(o.is_open) {
}

GazePointPersistor& GazePointPersistor::operator = (GazePointPersistor&& o) {
    gazes = std::move(o.gazes);
    converter = std::move(o.converter);
    is_open = std::move(o.is_open);
    framerate = std::move(o.framerate);
    return *this;
}

GazePointPersistor::GazePointPersistor(const std::string& filename, uint64_t timeStamp, double framerate, std::function<GazePoint (const GazePoint&)> convert) 
    : converter(convert), framerate(framerate) {
    if (!converter) {
        converter = +[](const GazePoint& p) {
            return p;
        };
    }

    read(filename, timeStamp);
}

void GazePointPersistor::close() {
    is_open=false;
}

GazePointPersistor GazePointPersistor::open(const std::string& filename, size_t timestamp, double framerate, std::function<GazePoint (const GazePoint&)> converter) {
    return std::move(GazePointPersistor(filename, timestamp, framerate, converter));
}

void GazePointPersistor::read(const std::string& filename, uint64_t timeStamp) {
    CSVRow row;
    std::ifstream stream(filename);
    if (!stream.is_open()) {
        std::cout << "Can not open: " << filename << std::endl;
        exit(-1);
        return;
    }

    is_open = true;
    stream >> row; // read header
    while (stream >> row) {
        if (!row) continue;

        auto timestamp = std::stoul(std::string(row[0]));
        if (timeStamp <= timestamp) {
            gazes.push_back(converter(GazePoint {
                static_cast<float>(std::stod(row[1])),   
                static_cast<float>(std::stod(row[2])),   
            }));
        }
    }
}
