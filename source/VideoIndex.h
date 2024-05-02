#pragma once

struct VideoIndex {
    VideoIndex() = default;
    VideoIndex(uint16_t video, uint32_t frame, uint16_t descriptor)
        : frame(frame), video(video), descriptor(descriptor) {
    }

    uint32_t frame;
    uint16_t video;
    uint16_t descriptor;

    inline bool operator == (const VideoIndex& o) const { return frame == o.frame && video == o.video; }
    inline bool operator != (const VideoIndex& o) const { return frame != o.frame || video != o.video; }

    bool operator < (const VideoIndex& o) const {
        return video != o.video ? video < o.video : (frame != o.frame ? frame < o.frame : descriptor < o.descriptor);
    }
};
